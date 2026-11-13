"""Depth, position, date, temperature and size fraction for the marine samples.

    python marine_metadata.py

Marine samples carry no depth or size-fraction labels in the course data, the variables that
structure plankton most. This step fetches them, resumable, from public records:
  1 NCBI BioSample (E-utilities: esearch by accession, efetch XML, batches of 100, <= 3 requests/s)
  2 ENA sample records (ebi.ac.uk/ena/browser/api/xml) for samples whose BioSample record lacks
    a depth (e.g. Tara Oceans, whose metadata are stored with ENA)
Parsed fields: depth_m (numeric, first number of the depth field; ranges use their midpoint),
lat, lon, date, temp_c, size_fraction (text). Raw XML is cached in data/motusdb/biosample_xml/.

-> data/motusdb/marine_metadata.tsv   one row per marine sample
   results/marine_metadata_coverage.tsv
"""
import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "P9-cooccurrence-analysis/1.0 (python urllib)"}
EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
DEPTH_KEYS = ["depth", "geographic location (depth)", "sampling depth", "water depth", "depth (m)", "sample depth"]
SIZE_KEYS = ["size_frac", "size fraction", "size_fraction", "filter size", "size_frac_low", "size fraction lower threshold",
             "size_frac_up", "size fraction upper threshold"]


def fetch(url, tries=4, sleep=0.35):
    for k in range(tries):
        try:
            time.sleep(sleep)
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except Exception:
            time.sleep(3 * (k + 1))
    return b""


def num(txt):
    if txt is None:
        return np.nan
    v = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(txt).replace(",", "."))]
    if not v:
        return np.nan
    return (v[0] + v[1]) / 2 if len(v) >= 2 and re.search(r"\d\s*-\s*\d", str(txt)) else v[0]


def latlon(txt):
    m = re.findall(r"(-?\d+(?:\.\d+)?)\s*([NSEW])?", str(txt or ""))
    if len(m) < 2:
        return np.nan, np.nan
    la, lo = float(m[0][0]), float(m[1][0])
    if m[0][1] == "S":
        la = -abs(la)
    if m[1][1] == "W":
        lo = -abs(lo)
    return la, lo


def parse_attrs(attrs):
    a = {k.lower().strip(): v for k, v in attrs.items()}
    out = {"depth_raw": next((a[k] for k in DEPTH_KEYS if k in a), None)}
    out["depth_m"] = num(out["depth_raw"])
    ll = a.get("lat_lon") or a.get("geographic location (latitude and longitude)")
    la = a.get("latitude") or a.get("geographic location (latitude)") or a.get("lat")
    lo = a.get("longitude") or a.get("geographic location (longitude)") or a.get("lon")
    out["lat"], out["lon"] = latlon(ll) if ll else (num(la), num(lo))
    out["date"] = a.get("collection_date") or a.get("collection date") or a.get("event date/time")
    out["temp_c"] = num(a.get("temp") or a.get("temperature") or a.get("water temperature"))
    out["size_fraction"] = "|".join(str(a[k]) for k in SIZE_KEYS if k in a) or None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args(argv)
    S = pd.read_csv(f"{a.out}/samples.tsv", sep="\t", index_col=0)
    m = S[S.environment == "Saline"]
    acc = sorted(m.biosample.dropna().unique())
    cdir = os.path.join(a.data, "motusdb", "biosample_xml"); os.makedirs(cdir, exist_ok=True)
    print(f"{len(m)} marine samples, {len(acc)} BioSample accessions")

    # ---- 1 NCBI BioSample
    recs = {}
    for k in range(0, len(acc), 100):
        part = acc[k:k + 100]
        cache = os.path.join(cdir, f"ncbi_{k:05d}.xml")
        if not os.path.exists(cache):
            term = " OR ".join(f"{x}[accn]" for x in part)
            js = fetch(EU + "esearch.fcgi?" + urllib.parse.urlencode({"db": "biosample", "term": term, "retmax": 200, "retmode": "json"}))
            ids = json.loads(js or b"{}").get("esearchresult", {}).get("idlist", [])
            xml = fetch(EU + "efetch.fcgi?" + urllib.parse.urlencode({"db": "biosample", "id": ",".join(ids)}), sleep=0.35) if ids else b""
            open(cache, "wb").write(xml)
        xml = open(cache, "rb").read()
        if xml.strip():
            for b in ET.fromstring(xml).findall("BioSample"):
                attrs = {x.get("attribute_name") or x.get("harmonized_name") or "": (x.text or "") for x in b.iter("Attribute")}
                recs[b.get("accession")] = parse_attrs(attrs) | {"source": "NCBI"}
        if (k // 100) % 10 == 0:
            print(f"  NCBI {min(k + 100, len(acc))}/{len(acc)}")

    # ---- 2 ENA for records without depth
    need = [x for x in acc if (x not in recs) or np.isnan(recs[x]["depth_m"])]
    print(f"ENA lookup for {len(need)} samples without depth")
    # ENA portal search: many samples per POST request (the per-sample XML endpoint is too slow)
    fields = "sample_accession,description,depth,lat,lon,collection_date,temperature"
    need = [x for x in need if str(x).startswith("SAM")]      # ENA knows only SAMEA/SAMN/SAMD accessions

    def portal(part):
        """One POST; ENA rejects a whole batch with HTTP 400 if any accession is unknown, so split it."""
        body = urllib.parse.urlencode({"result": "sample", "includeAccessions": ",".join(part),
                                       "fields": fields, "format": "tsv"}).encode()
        for t in range(3):
            try:
                time.sleep(0.5)
                req = urllib.request.Request("https://www.ebi.ac.uk/ena/portal/api/search", data=body, headers=UA)
                with urllib.request.urlopen(req, timeout=180) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code == 400:
                    if len(part) == 1:
                        return b""
                    h = len(part) // 2
                    a1, a2 = portal(part[:h]), portal(part[h:])
                    lines = [l for l in (a1 + b"\n" + a2).splitlines() if l.strip()]
                    hdr = lines[0] if lines else b""
                    return b"\n".join([hdr] + [l for l in lines if l != hdr]) + b"\n" if lines else b""
                time.sleep(5 * (t + 1))
            except Exception:
                time.sleep(5 * (t + 1))
        return b""
    for k in range(0, len(need), 300):
        part = need[k:k + 300]
        cache = os.path.join(cdir, f"enaportal_v2_{k:05d}.tsv")
        if not os.path.exists(cache):
            data = portal(part)
            if data.strip():
                open(cache, "wb").write(data)
        if not os.path.exists(cache):
            continue
        E = pd.read_csv(cache, sep="\t", dtype=str)
        for row in E.itertuples():
            x = row.sample_accession
            desc = str(row.description) if isinstance(row.description, str) else ""
            dl = desc.lower()
            p_ = {"depth_raw": row.depth if isinstance(row.depth, str) else None, "depth_m": num(row.depth),
                  "lat": num(row.lat), "lon": num(row.lon),
                  "date": row.collection_date if isinstance(row.collection_date, str) else None,
                  "temp_c": num(row.temperature),
                  # Tara Oceans keeps layer and size fraction in the description, e.g.
                  # "Station7 - Deep Chlorophyl Maximum - 5-20 um - WGA"
                  "layer": ("surface" if "surface" in dl else "DCM" if "chlorophyl" in dl else
                            "mesopelagic" if "mesopelagic" in dl else None)}
            msz = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:µm|um|micron|Âµm)", desc)
            p_["size_fraction"] = f"{msz.group(1)}-{msz.group(2)}" if msz else None
            old = recs.get(x, {})
            recs[x] = {kk: (p_.get(kk) if (kk not in old or old[kk] is None or (isinstance(old[kk], float) and np.isnan(old[kk])))
                            else old[kk]) for kk in set(p_) | set(old)} | {"source": "NCBI+ENA" if old else "ENA"}
        print(f"  ENA {min(k + 300, len(need))}/{len(need)}")

    R = pd.DataFrame.from_dict(recs, orient="index")
    # clean depth: placeholders (e.g. 99999) and impossible values -> missing; negative depths -> absolute
    R["depth_m"] = pd.to_numeric(R.depth_m, errors="coerce").abs()
    R.loc[R.depth_m > 11000, "depth_m"] = np.nan
    out = m[["study", "biosample", "habitat_group"]].join(R, on="biosample")
    out.index.name = "sample"
    out.to_csv(f"{a.data}/motusdb/marine_metadata.tsv", sep="\t")
    if "layer" not in out:
        out["layer"] = None
    cov = pd.DataFrame({"field": ["depth_m", "layer (Tara)", "depth_m or layer", "lat", "date", "temp_c", "size_fraction"],
                        "samples_with_value": [int(out.depth_m.notna().sum()), int(out.layer.notna().sum()),
                                               int((out.depth_m.notna() | out.layer.notna()).sum()), int(out.lat.notna().sum()),
                                               int(out.date.notna().sum()), int(out.temp_c.notna().sum()), int(out.size_fraction.notna().sum())]})
    cov["share"] = cov.samples_with_value / len(out)
    cov.to_csv(f"{a.out}/marine_metadata_coverage.tsv", sep="\t", index=False, float_format="%.3f")
    print(cov.to_string(index=False))
    print(out.groupby("study").depth_m.apply(lambda v: f"{v.notna().mean():.0%} with depth").head(12).to_string())
    print(out.depth_m.describe().round(1).to_string())


if __name__ == "__main__":
    main()
