import glob
import os
import re

html_files = sorted(glob.glob("*.html"))
print(f"=== Deep Quality Audit of {len(html_files)} Pages ===")

all_clean = True

for f in html_files:
    with open(f, "r", encoding="utf-8") as fp:
        html = fp.read()

    errors = []

    # 1. Check title
    if "<title>" not in html or "</title>" not in html:
        errors.append("Missing <title> tag")

    # 2. Check doctype
    if not html.strip().startswith("<!DOCTYPE html>"):
        errors.append("Missing <!DOCTYPE html>")

    # 3. Check image references
    images = re.findall(r'src=["\'](assets/images/.*?)["\']', html)
    for img in images:
        if not os.path.exists(img):
            errors.append(f"Missing referenced image: {img}")

    # 4. Check getElementById calls vs DOM IDs
    js_ids = re.findall(r'document\.getElementById\(["\'](.*?)["\']\)', html)
    dom_ids = set(re.findall(r'id=["\'](.*?)["\']', html))
    for jid in js_ids:
        if jid not in dom_ids:
            errors.append(f"JS calls getElementById('{jid}'), but id='{jid}' is NOT in DOM!")

    # 5. Check mobile menu button & nav drawer
    if "mobileMenuBtn" in js_ids and "mobileMenuBtn" not in dom_ids:
        errors.append("mobileMenuBtn missing")
    if "mobileNav" in js_ids and "mobileNav" not in dom_ids:
        errors.append("mobileNav missing")

    # 6. Check openLightbox and lightboxModal
    if "openLightbox" in html:
        if "lightboxModal" not in dom_ids:
            errors.append("openLightbox referenced but lightboxModal id missing")
        if "lightboxImage" not in dom_ids:
            errors.append("openLightbox referenced but lightboxImage id missing")
        if "lightboxTitle" not in dom_ids:
            errors.append("openLightbox referenced but lightboxTitle id missing")

    if errors:
        print(f"[ERROR] {f}:")
        for e in errors:
            print(f"  - {e}")
        all_clean = False
    else:
        print(f"[OK] {f:<18} passed all DOM, JS ID, and asset integrity checks.")

if all_clean:
    print(f"\n[SUCCESS] Flawless! All {len(html_files)} pages passed 100% of DOM, JS, and asset audits.")
else:
    print("\n[FAIL] Rectifications needed.")
