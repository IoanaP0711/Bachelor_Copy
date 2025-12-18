#!/usr/bin/env python3
import argparse, ipaddress, math
import pandas as pd, numpy as np
EPS=1e-9
def pick(df, opts): return next((c for c in opts if c in df.columns), None)
def to_ts_seconds(x):
    if pd.isna(x): return np.nan
    try: return float(x)
    except: pass
    t=pd.to_datetime(x, utc=True, errors="coerce"); 
    return t.value/1e9 if not pd.isna(t) else np.nan
def is_internal(ip):
    try:
        ip=ipaddress.ip_address(str(ip))
        return int(ip.is_private or ip.is_loopback or ip.is_link_local)
    except: return 0
def parse_tcp_flags_suricata(v):
    s=str(v) if pd.notna(v) else "0"
    try: n=int(s,16) if s.lower().startswith("0x") else int(s)
    except: n=0
    bits={"SYN":0x02,"ACK":0x10,"FIN":0x01,"RST":0x04,"PSH":0x08,"URG":0x20,"ECE":0x40,"CWR":0x80}
    return {f"tcp_flag_{k}":int(bool(n & b)) for k,b in bits.items()}
def parse_tcp_flags_zeek(h):
    h=str(h) if pd.notna(h) else ""
    return {"tcp_flag_SYN":int(("S" in h) or ("H" in h)),
            "tcp_flag_ACK":int(("D" in h) or ("H" in h)),
            "tcp_flag_FIN":int("F" in h),
            "tcp_flag_RST":int("R" in h),
            "tcp_flag_PSH":int("P" in h),
            "tcp_flag_URG":int("U" in h),
            "tcp_flag_ECE":0,"tcp_flag_CWR":0}
def infer_source(df):
    if any(c.startswith("flow.") for c in df.columns) or "app_proto" in df.columns: return "suricata"
    if "history" in df.columns or "conn_state" in df.columns or "id.orig_h" in df.columns: return "zeek"
    return "suricata"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", required=True, dest="inp")
    ap.add_argument("--out", required=True, dest="outp")
    ap.add_argument("--source", default="auto", choices=["auto","suricata","zeek"])
    args=ap.parse_args()
    df=pd.read_csv(args.inp)
    src=args.source if args.source!="auto" else infer_source(df)
    src_ip=pick(df,["src_ip","id.orig_h"]); dst_ip=pick(df,["dst_ip","id.resp_h"])
    src_p=pick(df,["src_port","id.orig_p"]); dst_p=pick(df,["dst_port","id.resp_p"])
    proto=pick(df,["proto","transport"]); app=pick(df,["app_proto","service"])
    start=pick(df,["flow.start_ts","flow_start_ts","start","ts","timestamp"])
    end=pick(df,["flow.end_ts","flow_end_ts","end"]); dur=pick(df,["flow.duration","duration_s","duration"])
    p_srv=pick(df,["flow.pkts_toserver","pkts_toserver","orig_pkts"])
    p_cli=pick(df,["flow.pkts_toclient","pkts_toclient","resp_pkts"])
    b_srv=pick(df,["flow.bytes_toserver","bytes_toserver","orig_ip_bytes","orig_bytes"])
    b_cli=pick(df,["flow.bytes_toclient","bytes_toclient","resp_ip_bytes","resp_bytes"])
    tflags=pick(df,["tcp.flags","tcp_flags"]); zhist=pick(df,["history"]); zstate=pick(df,["conn_state"])
    w=pd.DataFrame(index=df.index)
    w["proto"]=df[proto].astype(str).str.upper() if proto else "UNK"
    w["app"]=df[app].astype(str).str.lower() if app else "unknown"
    w["src_ip"]=df[src_ip] if src_ip else ""; w["dst_ip"]=df[dst_ip] if dst_ip else ""
    w["src_port"]=pd.to_numeric(df[src_p],errors="coerce").fillna(0).astype(int) if src_p else 0
    w["dst_port"]=pd.to_numeric(df[dst_p],errors="coerce").fillna(0).astype(int) if dst_p else 0
    start_s=df[start].map(to_ts_seconds) if start else pd.Series(np.nan,index=df.index)
    end_s=df[end].map(to_ts_seconds) if end else pd.Series(np.nan,index=df.index)
    duration=pd.to_numeric(df[dur],errors="coerce") if dur else end_s-start_s
    w["duration_s"]=duration.fillna(0).clip(lower=0)
    def num(c): return pd.to_numeric(df[c],errors="coerce") if c else pd.Series(0,index=df.index)
    s_pkts=num(p_srv).fillna(0).clip(lower=0); c_pkts=num(p_cli).fillna(0).clip(lower=0)
    s_bytes=num(b_srv).fillna(0).clip(lower=0); c_bytes=num(b_cli).fillna(0).clip(lower=0)
    w["pkts_toserver"]=s_pkts; w["pkts_toclient"]=c_pkts
    w["bytes_toserver"]=s_bytes; w["bytes_toclient"]=c_bytes
    w["pkts_total"]=s_pkts+c_pkts; w["bytes_total"]=s_bytes+c_bytes
    d=w["duration_s"].clip(lower=EPS); pt=w["pkts_total"].clip(lower=1); bt=w["bytes_total"].clip(lower=1)
    w["pps"]=w["pkts_total"]/d; w["bps"]=8*w["bytes_total"]/d; w["avg_pkt_size"]=w["bytes_total"]/pt
    w["pkts_dir_ratio"]=s_pkts/c_pkts.clip(lower=1); w["bytes_dir_ratio"]=s_bytes/c_bytes.clip(lower=1)
    w["syn_asymmetry"]=(s_bytes-c_bytes)/bt; w["pkts_asymmetry"]=(s_pkts-c_pkts)/pt
    p_srv_b=(s_bytes/bt).astype(float); p_cli_b=(c_bytes/bt).astype(float)
    w["dir_entropy"]=(-p_srv_b*np.log2(p_srv_b.where(p_srv_b>0,1)) + -p_cli_b*np.log2(p_cli_b.where(p_cli_b>0,1))).fillna(0)
    w["server_pps"]=s_pkts/d; w["client_pps"]=c_pkts/d
    w["server_bps"]=8*s_bytes/d; w["client_bps"]=8*c_bytes/d
    w["bytes_per_server_pkt"]=s_bytes/s_pkts.clip(lower=1); w["bytes_per_client_pkt"]=c_bytes/c_pkts.clip(lower=1)
    w["l4_port_class_src"]=(w["src_port"]<1024).astype(int); w["l4_port_class_dst"]=(w["dst_port"]<1024).astype(int)
    w["is_internal_src"]=[is_internal(x) for x in w["src_ip"]]; w["is_internal_dst"]=[is_internal(x) for x in w["dst_ip"]]
    w["pat_int_int"]=((w["is_internal_src"]==1)&(w["is_internal_dst"]==1)).astype(int)
    w["pat_int_ext"]=((w["is_internal_src"]==1)&(w["is_internal_dst"]==0)).astype(int)
    w["pat_ext_int"]=((w["is_internal_src"]==0)&(w["is_internal_dst"]==1)).astype(int)
    w["pat_ext_ext"]=((w["is_internal_src"]==0)&(w["is_internal_dst"]==0)).astype(int)
    if start:
        ts=pd.to_datetime(start_s,unit="s",utc=True); sec=ts.dt.hour*3600+ts.dt.minute*60+ts.dt.second
        th=2*math.pi*(sec/86400.0); w["time_of_day_sin"]=np.sin(th); w["time_of_day_cos"]=np.cos(th)
    else:
        w["time_of_day_sin"]=0.0; w["time_of_day_cos"]=0.0
    for p in ["TCP","UDP","ICMP"]: w[f"proto_{p}"]=(w["proto"]==p).astype(int)
    for k in ["tcp_flag_SYN","tcp_flag_ACK","tcp_flag_FIN","tcp_flag_RST","tcp_flag_PSH","tcp_flag_URG","tcp_flag_ECE","tcp_flag_CWR"]: w[k]=0
    if src=="suricata" and tflags: w[list(parse_tcp_flags_suricata(0).keys())]=df[tflags].apply(parse_tcp_flags_suricata).apply(pd.Series).astype(int)
    if src=="zeek" and zhist: w[list(parse_tcp_flags_zeek("").keys())]=df[zhist].apply(parse_tcp_flags_zeek).apply(pd.Series).astype(int)
    if zstate and zstate in df.columns: 
        st=df[zstate].astype(str); w["tcp_handshake_ok"]=st.isin(["S1","SF"]).astype(int)
    else: w["tcp_handshake_ok"]=0
    for p in [80,443,53,123,22,25,3389,445,5353,8080,8000]: w[f"dstp_{p}"]=(w["dst_port"]==p).astype(int)
    w["dstp_other"]=(~w["dst_port"].isin([80,443,53,123,22,25,3389,445,5353,8080,8000])).astype(int)
    w["small_flow_flag"]=((w["pkts_total"]<=3)|(w["bytes_total"]<=300)).astype(int)
    w["long_flow_flag"]=(w["duration_s"]>=600).astype(int)
    out=w.drop(columns=["src_ip","dst_ip","src_port","dst_port","proto","app"], errors="ignore").replace([np.inf,-np.inf],np.nan).fillna(0)
    for c in ["bps","pps","server_bps","client_bps"]:
        if c in out.columns: out[c]=out[c].clip(upper=out[c].quantile(0.999, interpolation="nearest"))
    out.to_csv(args.outp, index=False)
    print(f"[OK] Features: {out.shape[1]} cols, {out.shape[0]} rows -> {args.outp}")
if __name__=="__main__": main()
