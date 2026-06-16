# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Inline stylesheet for the HTML report (no external requests)."""

# Severity tokens follow the real diagnosis vocabulary crit | warn | info
# (diagnostics/common.py); the info tier is split by kind into a green
# "healthy" and a neutral "no data" presentation in the renderer.
STYLESHEET = """
  :root {
    --bg:#f6f7f9; --card:#fff; --ink:#1c2430; --muted:#5d6b7e;
    --line:#e3e7ed; --chip:#eef1f5;
    --crit:#c4392f; --crit-bg:#fdf0ef; --warn:#b97a14; --warn-bg:#fdf6e9;
    --info:#2f6db8; --info-bg:#eef4fb; --good:#2e7d52; --good-bg:#edf7f1;
    --neutral:#6b7684; --neutral-bg:#f1f3f5;
    --dl:#d97742; --h2d:#8e6fd8; --fwd:#3b82c4; --bwd:#1e5a96;
    --opt:#3aa17e; --wait:#98a2ae;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg:#14181f; --card:#1c222c; --ink:#e6eaf0; --muted:#98a4b5;
      --line:#2b3340; --chip:#252d39;
      --crit:#e06158; --crit-bg:#34201f; --warn:#d9a046; --warn-bg:#322a1a;
      --info:#6ea4dd; --info-bg:#1d2938; --good:#5cb585; --good-bg:#1c2e24;
      --neutral:#8d99a8; --neutral-bg:#232932;
    }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
    Helvetica,Arial,sans-serif; }
  .wrap { max-width:980px; margin:0 auto; padding:28px 20px 48px; }
  header.run { display:flex; flex-wrap:wrap; align-items:baseline;
    gap:10px 16px; margin-bottom:6px; }
  header.run h1 { font-size:22px; margin:0; font-weight:650; }
  .logo { font-weight:750; } .logo span { color:var(--info); }
  .meta-chips { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 22px; }
  .chip { background:var(--chip); border:1px solid var(--line);
    border-radius:6px; padding:3px 10px; font-size:13px; color:var(--muted); }
  .chip b { color:var(--ink); font-weight:600; }
  .banner { border:1px solid var(--line); border-radius:8px;
    padding:16px 18px; margin-bottom:26px; }
  .banner.crit { border-left:6px solid var(--crit); background:var(--crit-bg);}
  .banner.warn { border-left:6px solid var(--warn); background:var(--warn-bg);}
  .banner.good { border-left:6px solid var(--good); background:var(--good-bg);}
  .banner.neutral { border-left:6px solid var(--neutral);
    background:var(--neutral-bg); }
  .banner .kind { display:inline-block; font-size:12px; font-weight:700;
    letter-spacing:.6px; color:#fff; border-radius:4px; padding:2px 8px;
    margin-right:8px; vertical-align:2px; }
  .banner.crit .kind { background:var(--crit);}
  .banner.warn .kind { background:var(--warn);}
  .banner.good .kind { background:var(--good);}
  .banner.neutral .kind { background:var(--neutral);}
  .banner h2 { display:inline; font-size:17px; margin:0; font-weight:650; }
  .banner p { margin:8px 0 0; }
  .banner .action { color:var(--muted); } .banner .action b { color:var(--ink);}
  section.card { background:var(--card); border:1px solid var(--line);
    border-radius:8px; padding:18px 20px; margin-bottom:22px; }
  section.card > h3 { margin:0 0 4px; font-size:16px; font-weight:650; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:12px; }
  .diag { border-left:4px solid var(--neutral); background:var(--neutral-bg);
    border-radius:6px; padding:10px 14px; margin:10px 0 14px; }
  .diag.crit { border-color:var(--crit); background:var(--crit-bg); }
  .diag.warn { border-color:var(--warn); background:var(--warn-bg); }
  .diag.good { border-color:var(--good); background:var(--good-bg); }
  .diag.info { border-color:var(--info); background:var(--info-bg); }
  .badge { display:inline-block; font-size:11px; font-weight:700;
    letter-spacing:.5px; border-radius:4px; padding:1px 7px; color:#fff;
    margin-right:8px; vertical-align:1px; }
  .badge.crit{background:var(--crit);} .badge.warn{background:var(--warn);}
  .badge.good{background:var(--good);} .badge.info{background:var(--info);}
  .badge.neutral{background:var(--neutral);}
  .diag .why { color:var(--muted); font-size:13px; margin-top:4px; }
  .tablewrap { overflow-x:auto; }
  table { border-collapse:collapse; width:100%; font-size:13.5px; }
  th,td { text-align:right; padding:6px 10px;
    border-bottom:1px solid var(--line); white-space:nowrap; }
  th { color:var(--muted); font-weight:600; }
  th:first-child, td:first-child { text-align:left; }
  td.metric { text-align:left; color:var(--muted); }
  .idx { color:var(--muted); font-size:12px; }
  .num { font-variant-numeric:tabular-nums; }
  details { margin-top:12px; }
  summary { cursor:pointer; color:var(--info); font-size:13.5px;
    font-weight:600; }
  details[open] summary { margin-bottom:8px; }
  .legend { display:flex; flex-wrap:wrap; gap:6px 16px; margin-top:8px;
    font-size:12.5px; color:var(--muted); }
  .sw { display:inline-block; width:10px; height:10px; border-radius:2px;
    margin-right:5px; vertical-align:-1px; }
  .cellbar { display:flex; align-items:center; gap:8px;
    justify-content:flex-end; }
  .cellbar .track { width:84px; height:7px; border-radius:4px;
    background:var(--chip); overflow:hidden; flex:none; }
  .cellbar .fill { height:100%; background:var(--info); }
  .membars { margin:6px 0 2px; }
  .membar { display:grid; grid-template-columns:200px 1fr 92px;
    align-items:center; gap:12px; margin:4px 0; font-size:13px; }
  .membar .track { height:8px; border-radius:4px; background:var(--chip);
    overflow:hidden; }
  .membar .fill { height:100%; background:var(--info); }
  .membar .num { text-align:right; font-variant-numeric:tabular-nums; }
  pre.raw { background:var(--chip); border:1px solid var(--line);
    border-radius:6px; padding:12px 14px; overflow-x:auto; font-size:12px;
    line-height:1.45; }
  footer { color:var(--muted); font-size:12.5px; text-align:center;
    margin-top:30px; }
"""

__all__ = ["STYLESHEET"]
