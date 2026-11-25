#!/usr/bin/env python3
"""
PhytoGuardDL Automated Python Web Services Client
=================================================
Nucleic Acids Research (NAR) Web Server Issue Reference Client
Supports: /health, /api/predict (single), /api/batch (FASTA), /api/job/{id}

Usage:
  python phytoguarddl_client.py --health
  python phytoguarddl_client.py --sequence "MKLLSKTFLILTLTFFFFGIALAKQSF..." --name "FLS2_AT5G46330"
  python phytoguarddl_client.py --fasta examples/sample_proteins.fasta --out predictions.tsv
  python phytoguarddl_client.py --job pg_job_xxxxxxxx
"""
import argparse
import csv
import json
import urllib.request

DEFAULT_URL = "https://phytoguarddl.cdfd.org.in"


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def write_tsv(results: list, out_path: str) -> None:
    cols = [
        "name", "length", "prediction", "biological_axis", "confidence",
        "ensemble_score", "threshold", "phase2_class", "subcellular_localization",
        "phytohormone_pathway", "go_terms", "kegg_pathways", "pfam_domains",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        for r in results:
            p2 = r.get("phase2") or {}
            w.writerow([
                r.get("name", "query"), r.get("length", 0), r.get("prediction", "N/A"),
                r.get("biological_axis", p2.get("biological_axis", "Basal")),
                r.get("confidence", "N/A"), r.get("ensemble_score", 0.0),
                r.get("threshold", 0.3731), p2.get("phase2_class", "N/A"),
                p2.get("subcellular_localization", "N/A"),
                p2.get("phytohormone_pathway", "N/A"),
                "; ".join(p2.get("go_terms", [])),
                "; ".join(p2.get("kegg_pathways", [])),
                "; ".join(p2.get("pfam_domains", [])),
            ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PhytoGuardDL REST API Python Client")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"Base server URL (default: {DEFAULT_URL})")
    ap.add_argument("--health", action="store_true", help="Check server health status")
    ap.add_argument("--sequence", help="Single protein amino acid sequence")
    ap.add_argument("--name", default="query", help="Protein identifier")
    ap.add_argument("--fasta", help="Path to multi-FASTA file")
    ap.add_argument("--job", help="Retrieve a bookmarked batch Job ID")
    ap.add_argument("--out", help="Optional output TSV path")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    if args.health:
        print(json.dumps(get_json(f"{base}/health"), indent=2))
    elif args.job:
        res = get_json(f"{base}/api/job/{args.job}")
        print(f"Job {args.job} | Status: {res.get('status', 'finished')} | Total: {res.get('total', len(res.get('results', [])))}")
        if args.out and res.get("results"):
            write_tsv(res["results"], args.out)
            print(f"Saved parsed output to {args.out}")
    elif args.sequence:
        res = post_json(f"{base}/api/predict", {"sequence": args.sequence, "name": args.name})
        p2 = res.get("phase2") or {}
        print(f"Name: {res.get('name')} ({res.get('length')} aa)")
        print(f"Prediction: {res.get('prediction')} | Axis: {res.get('biological_axis')} | Score: {res.get('ensemble_score')}")
        print(f"Functional Class: {p2.get('phase2_class')} | Localization: {p2.get('subcellular_localization')}")
        if args.out:
            write_tsv([res], args.out)
            print(f"Saved parsed output to {args.out}")
    elif args.fasta:
        with open(args.fasta, "r", encoding="utf-8") as f:
            fasta_txt = f.read()
        res = post_json(f"{base}/api/batch", {"fasta": fasta_txt})
        print(f"Total: {res['total']} | Stress/Defense: {res['defense']} | Basal: {res['non_defense']}")
        print(f"Job ID: {res.get('job_id')} | Bookmark: {base}/?job={res.get('job_id')}")
        if args.out and res.get("results"):
            write_tsv(res["results"], args.out)
            print(f"Saved parsed output to {args.out}")
    else:
        ap.print_help()
