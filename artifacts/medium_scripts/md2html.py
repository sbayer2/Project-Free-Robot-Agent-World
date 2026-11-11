import html
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src, encoding="utf-8").read().split("\n")


def inline(t):
    codes = []

    def stash(m):
        codes.append(html.escape(m.group(1), quote=False))
        return f"\x00{len(codes) - 1}\x00"

    t = re.sub(r"`([^`]+)`", stash, t)
    t = html.escape(t, quote=False)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", t)
    t = re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{codes[int(m.group(1))]}</code>", t)
    return t


out, para, inlist, incode, code = [], [], False, False, []
pending_img = None


def flush():
    global para
    if para:
        out.append("<p>" + inline(" ".join(para)) + "</p>")
        para = []


for ln in lines:
    if incode:
        if ln.startswith("```"):
            out.append('<pre><code>' + html.escape("\n".join(code), quote=False) + "</code></pre>")
            incode, code = False, []
        else:
            code.append(ln)
        continue
    if ln.startswith("```"):
        flush()
        if inlist:
            out.append("</ol>"); inlist = False
        incode = True
        continue
    img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", ln.strip())
    m = re.match(r"^(#{1,3}) (.*)$", ln)
    n = re.match(r"^\d+\. (.*)$", ln)
    if img:
        flush()
        pending_img = (img.group(1), img.group(2))
        continue
    if pending_img and ln.strip().startswith("*Figure"):
        alt, path = pending_img
        cap = ln.strip().strip("*")
        out.append(f'<figure><img src="{path}" alt="{html.escape(alt)}"><figcaption>{inline(cap)}</figcaption></figure>')
        pending_img = None
        continue
    if pending_img and ln.strip() == "":
        continue
    if m:
        flush()
        if inlist:
            out.append("</ol>"); inlist = False
        lvl = len(m.group(1)); txt = m.group(2)
        out.append(f"<h{lvl}>{inline(txt)}</h{lvl}>" if lvl < 3 else f'<p class="subtitle">{inline(txt)}</p>')
    elif n:
        flush()
        if not inlist:
            out.append("<ol>"); inlist = True
        out.append("<li>" + inline(n.group(1)) + "</li>")
    elif ln.strip() == "":
        flush()
        if inlist:
            out.append("</ol>"); inlist = False
    else:
        para.append(ln.strip())
flush()
if inlist:
    out.append("</ol>")

css = """
:root{--ink:#1f2328;--soft:#57606a;--rule:#d8dee4;--paper:#ffffff;--accent:#1a5fb4;--codebg:#f6f8fa}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:19px/1.65 Georgia,'Iowan Old Style','Times New Roman',serif}
main{max-width:700px;margin:0 auto;padding:48px 20px 80px}
h1{font:700 2.1rem/1.2 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;margin:0 0 12px;letter-spacing:-.01em}
.subtitle{font:400 1.15rem/1.5 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;color:var(--soft);margin:0 0 36px}
h2{font:700 1.4rem/1.3 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;margin:44px 0 12px}
p{margin:0 0 18px}
a{color:var(--accent)}
ol{margin:0 0 20px;padding-left:1.4em}
li{margin:0 0 10px}
strong{font-weight:700}
em{color:var(--soft)}
code{font:0.86em ui-monospace,Menlo,Consolas,monospace;background:var(--codebg);padding:.1em .3em;border-radius:4px}
pre{background:var(--codebg);border:1px solid var(--rule);border-radius:6px;padding:14px 16px;overflow-x:auto;margin:0 0 22px}
pre code{background:none;padding:0;font-size:14.5px;line-height:1.5}
figure{margin:28px 0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--rule);border-radius:4px}
figcaption{font:14.5px/1.5 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;color:var(--soft);margin-top:8px}
.note{font:14px/1.5 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;color:var(--soft);border-top:1px solid var(--rule);padding-top:12px;margin-top:40px}
@media print{body{font-size:11.5pt;color:#000}main{max-width:none;padding:0}h2{break-after:avoid}li,p,figure,pre{break-inside:avoid}.note{display:none}}
"""
doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>pseudo-marble Medium draft</title><style>{css}</style></head>
<body><main>
{chr(10).join(out)}
<p class="note">Draft for Medium. Source: PSEUDO_MARBLE_MEDIUM.md. Text, headings, lists, links and code blocks paste into the Medium editor; figures do not, so upload each PNG from medium_figs/ at its marked place and paste its caption underneath.</p>
</main></body></html>
"""
open(dst, "w", encoding="utf-8").write(doc)
print("ok", len(out), "blocks;", sum(1 for o in out if o.startswith("<figure")), "figures;",
      sum(1 for o in out if o.startswith("<pre")), "code blocks")
