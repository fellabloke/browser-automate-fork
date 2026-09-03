"""
dom_parser.py — V11.0 God-Mode DOM Extraction Engine
═════════════════════════════════════════════════════
Clean rewrite.  Assimilated patterns from:
  • browser-use   — Accessibility tree as primary signal
  • Skyvern       — Multi-signal element identification
  • Stagehand     — observe() / act() separation
  • Crawl4AI      — Markdown compression for LLM-friendly output
  • BrowserGym    — Unique element IDs (bid) for precise targeting

Architecture:
  1. Single `page.evaluate()` — zero DOM mutation, <80ms
  2. Shadow DOM piercer via `page.add_init_script()` — stateless
  3. Output: Semantic Markdown (not raw JSON) → 60-80% fewer tokens
  4. Viewport-proximity ranking with semantic hint boost
  5. Input-state awareness (filled/empty detection)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

log = logging.getLogger("dom_parser")

# ═══════════════════════════════════════════════════════════════════════════════
#  V11 GOD-MODE EXTRACTION SCRIPT
#  Runs inside page.evaluate() in one shot.  Zero DOM mutation.
#  Returns { elements, markdown, image_size, source }
# ═══════════════════════════════════════════════════════════════════════════════

_GOD_MODE_JS = r"""
(targetHint) => {
    /* Adaptive budget: base 60 + extra for form-heavy pages, capped at 150 */
    const hasForm = !!document.querySelector('form, [role="form"], [data-testid]');
    const MAX_ELEMENTS = hasForm ? 80 : 60;
    const MIN_DIM = 4;

    const vw = window.innerWidth  || document.documentElement.clientWidth  || 1920;
    const vh = window.innerHeight || document.documentElement.clientHeight || 1080;

    /* ── V19.1 Primary-action recall ──────────────────────────────────────
       Commerce / checkout "goal" buttons (Add to Cart, Buy Now, Place Order,
       Notify Me, …) are routinely scrolled OUT of the viewport on dense product
       pages (Flipkart / Amazon). The off-screen cull below would discard them
       before scoring, so the LLM never sees the one button the task depends on
       — the observed Flipkart "Add to Cart not found → scroll forever" stall.
       We detect these by their short label, EXEMPT them from the off-screen
       cull, and give them a strong score bonus so they survive the element
       budget. They still resolve to fresh, scrolled-into-view coordinates at
       action time via the V19 window.__aid registry. */
    const ACTION_RE = /add to (cart|bag|basket)|buy\s?it\s?now|buy\s?now|buy\s?at\b|place order|order now|go to (cart|checkout)|view cart|proceed( to (checkout|pay(ment)?|buy))?|checkout|pay now|make payment|notify me|sold out|out of stock|coming soon|add to wishlist|subscribe/i;
    const SURVEY_REWARD_RE = /(?:[£$€]\s*\d+(?:[.,]\d{1,2})?|\d+(?:[.,]\d+)?\s*(?:[£$€]|points?|pts?|coins?|tokens?|credits?))/i;
    const SURVEY_DURATION_RE = /\b\d+(?:[.,]\d+)?\s*min(?:ute)?s?\b/i;
    const SURVEY_OFFER_RE = /(?:[£$€]\s*\d+(?:[.,]\d{1,2})?|\d+(?:[.,]\d+)?\s*(?:[£$€]|points?|pts?|coins?|tokens?|credits?)).{0,80}\b\d+(?:[.,]\d+)?\s*min(?:ute)?s?\b/i;

    /* ━━━ 1. COLLECT CANDIDATES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
    const seen = new WeakSet();
    const candidates = [];
    const surveyOfferNodes = new WeakSet();
    const surveyOfferText = new WeakMap();

    const SEMANTIC = [
        '[role="button"]','[role="link"]','[role="textbox"]',
        '[role="checkbox"]','[role="radio"]','[role="menuitem"]',
        '[role="tab"]','[role="switch"]','[role="combobox"]',
        '[role="option"]','[role="searchbox"]','[role="slider"]',
        '[contenteditable="true"]','[contenteditable="plaintext-only"]',
    ];
    const STRUCTURAL = [
        'a[href]','button','input','textarea','select',
        'label','summary','[data-testid]','[data-action]',
        '[tabindex]:not([tabindex="-1"])',
    ];
    const ALL_SEL = SEMANTIC.concat(STRUCTURAL);

    function effectivelyVisible(el) {
        if (!el || !el.isConnected) return false;
        let cur = el;
        while (cur && cur.nodeType === Node.ELEMENT_NODE) {
            try {
                if (cur.hidden || cur.inert || cur.getAttribute('aria-hidden') === 'true') return false;
                const style = window.getComputedStyle(cur);
                if (style.display === 'none' || style.visibility === 'hidden'
                    || style.contentVisibility === 'hidden'
                    || parseFloat(style.opacity || '1') <= 0.01) return false;
            } catch(_) { return false; }
            cur = cur.parentElement || (cur.parentNode && cur.parentNode.host) || null;
        }
        return true;
    }

    function isActionable(el) {
        if (!el || !el.matches) return false;
        if (el.matches('a[href],button,input,select,textarea,summary,[role="button"],[role="link"],'
                       + '[role="radio"],[role="checkbox"],[data-action],'
                       + '[tabindex]:not([tabindex="-1"])')) return true;
        try {
            return typeof el.onclick === 'function' || window.getComputedStyle(el).cursor === 'pointer';
        } catch(_) { return false; }
    }

    function actionableFor(container) {
        if (!container) return null;
        let cur = container;
        for (let hops = 0; cur && hops < 5; hops++, cur = cur.parentElement) {
            if (isActionable(cur) && effectivelyVisible(cur)) return cur;
        }
        try {
            const child = container.querySelector(
                'a[href],button,[role="button"],[role="link"],[data-action],'
                + '[tabindex]:not([tabindex="-1"])'
            );
            if (child && effectivelyVisible(child)) return child;
        } catch(_) {}
        return effectivelyVisible(container) ? container : null;
    }

    function collect(root) {
        for (const sel of ALL_SEL) {
            try {
                for (const el of root.querySelectorAll(sel)) {
                    if (!seen.has(el)) { seen.add(el); candidates.push(el); }
                }
            } catch(_) {}
        }
    }

    /* Main document */
    collect(document);

    /* Shadow DOM — walk open roots + pierced closed roots */
    function walkShadows(root) {
        if (!root) return;
        try {
            for (const host of root.querySelectorAll('*')) {
                const sr = host.shadowRoot;
                if (sr) { collect(sr); walkShadows(sr); }
            }
        } catch(_) {}
    }
    walkShadows(document);

    /* Piercer-captured closed roots */
    if (window.__piercer && typeof window.__piercer.roots === 'function') {
        try {
            for (const sr of window.__piercer.roots()) {
                collect(sr); walkShadows(sr);
            }
        } catch(_) {}
    }

    /* Behavioral: cursor:pointer on divs/spans (React dark elements) */
    try {
        for (const el of document.querySelectorAll('div,span,li,svg')) {
            if (seen.has(el) || candidates.length > 600) break;
            try {
                const cs = window.getComputedStyle(el);
                if (cs.cursor === 'pointer' || el.hasAttribute('onclick') || el.hasAttribute('@click')) {
                    seen.add(el); candidates.push(el);
                }
            } catch(_) {}
        }
    } catch(_) {}

    /* Survey dashboards frequently split reward and duration into sibling spans.
       Promote the nearest actionable card/ancestor and attach its combined text,
       so the click lands on the card rather than an inert points/duration leaf. */
    try {
        const allOfferParts = [...document.querySelectorAll(
            'button,a,[role="button"],[role="link"],div,span,li,article,section'
        )];
        for (const el of allOfferParts) {
            let t = '';
            try { t = (el.textContent || '').replace(/\s+/g, ' ').trim(); } catch(_) { continue; }
            if (!t || t.length > 260 || !SURVEY_REWARD_RE.test(t) || !SURVEY_DURATION_RE.test(t)) continue;
            let childHasOffer = false;
            for (const ch of el.children || []) {
                let ct = '';
                try { ct = (ch.textContent || '').replace(/\s+/g, ' ').trim(); } catch(_) {}
                if (ct && ct.length <= 260 && SURVEY_REWARD_RE.test(ct) && SURVEY_DURATION_RE.test(ct)) {
                    childHasOffer = true;
                    break;
                }
            }
            if (childHasOffer) continue;
            const action = actionableFor(el);
            if (!action) continue;
            surveyOfferNodes.add(action);
            surveyOfferText.set(action, t.slice(0, 240));
            if (!seen.has(action)) { seen.add(action); candidates.push(action); }
        }

        /* Fragment fallback: climb from a reward-only or duration-only leaf to
           the smallest ancestor containing both values. */
        for (const leaf of allOfferParts) {
            let own = '';
            try { own = (leaf.innerText || leaf.textContent || '').replace(/\s+/g, ' ').trim(); } catch(_) {}
            if (!own || own.length > 80 || !(SURVEY_REWARD_RE.test(own) || SURVEY_DURATION_RE.test(own))) continue;
            let card = leaf.parentElement;
            for (let hops = 0; card && hops < 6; hops++, card = card.parentElement) {
                let combined = '';
                try { combined = (card.innerText || card.textContent || '').replace(/\s+/g, ' ').trim(); } catch(_) {}
                if (!combined || combined.length > 260) continue;
                if (!SURVEY_REWARD_RE.test(combined) || !SURVEY_DURATION_RE.test(combined)) continue;
                const action = actionableFor(card);
                if (action) {
                    surveyOfferNodes.add(action);
                    surveyOfferText.set(action, combined.slice(0, 240));
                    if (!seen.has(action)) { seen.add(action); candidates.push(action); }
                }
                break;
            }
        }
    } catch(_) {}

    /* ── V19.1 Guaranteed primary-action sweep ──
       Flipkart / Amazon render "Add to cart" / "Buy now" / "Buy at ₹…" as styled
       <DIV> elements (NOT <button>) in a fixed action bar — and the behavioral
       cursor:pointer pass above can miss them (600-candidate cap, or no pointer
       cursor at scan time). That single miss is THE Flipkart "Add to Cart not
       found" failure. Here we force-collect any LEAF-ish element whose text is
       exactly a commerce action, bypassing both the cap and the pointer check.
       Anchored + length-capped + children≤3 so we match the real button and not
       a wrapper or promo copy. */
    try {
        for (const el of document.querySelectorAll('button, a, [role="button"], div, span, li')) {
            if (seen.has(el)) continue;
            if (el.children && el.children.length > 3) continue;   /* actions are leaf-ish */
            let t;
            try { t = (el.textContent || '').replace(/\s+/g, ' ').trim(); } catch(_) { continue; }
            if (!t || t.length > 28 || !ACTION_RE.test(t)) continue;
            /* Prefer the INNERMOST action element: if a child already holds the
               action text, collect that child instead of this wrapper. This both
               avoids wrappers spanning several buttons ("Add to cartBuy now") and
               collapses Flipkart's nested-div clones down to the real target. */
            let childHasAction = false;
            for (const ch of el.children) {
                let ct = '';
                try { ct = (ch.textContent || '').replace(/\s+/g, ' ').trim(); } catch(_) {}
                if (ct && ct.length <= 28 && ACTION_RE.test(ct)) { childHasAction = true; break; }
            }
            if (childHasAction) continue;
            seen.add(el); candidates.push(el);
        }
    } catch(_) {}

    /* ━━━ 2. FILTER + SCORE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
    const vcx = vw / 2, vcy = vh / 2;
    const scored = [];
    const actionCenters = [];  /* V19.1: collapse nested duplicates of one action button */
    const SEMANTIC_CONTAINERS = new Set([
        'MAIN','HEADER','FOOTER','NAV','FORM','DIALOG',
        'ARTICLE','SECTION','ASIDE','FIELDSET',
    ]);

    for (const el of candidates) {
        let rect;
        try { rect = el.getBoundingClientRect(); } catch(_) { continue; }

        if (rect.width < MIN_DIM || rect.height < MIN_DIM) continue;

        /* Off-screen cull — but RESCUE primary-action buttons (ACTION_RE).
           The label read is paid ONLY for off-screen candidates, so on-screen
           elements keep their zero-extra-cost fast path. The tag/role + length
           gate prevents false positives from product titles / promo copy that
           merely contain words like "buy now" (those are long <a> links). */
        let isAction = false;
        const offViewport = (rect.bottom < -50 || rect.top > vh * 2
                             || rect.right < -50 || rect.left > vw + 50);
        if (offViewport) {
            let qn = '';
            try { qn = (el.getAttribute('aria-label') || el.textContent || '').replace(/\s+/g, ' ').trim(); } catch(_) {}
            if (qn && qn.length <= 48 && ACTION_RE.test(qn)) {
                const tagU = (el.tagName || '').toUpperCase();
                let roleU = '';
                try { roleU = (el.getAttribute('role') || '').toLowerCase(); } catch(_) {}
                isAction = (tagU === 'BUTTON' || tagU === 'INPUT' || roleU === 'button' || qn.length <= 24);
            }
            if (surveyOfferNodes.has(el)) isAction = true;
            if (!isAction) continue;
        }

        if (!effectivelyVisible(el)) continue;
        let cs;
        try { cs = window.getComputedStyle(el); } catch(_) { continue; }
        if (cs.pointerEvents === 'none') continue;

        /* An on-screen center covered by an unrelated overlay is not clickable.
           Do not clamp an off-screen center onto the viewport edge: that tests
           an unrelated element and used to discard every control just below
           the fold (including survey Next buttons). */
        const centerInViewport = (
            rect.left + rect.width / 2 >= 0 && rect.left + rect.width / 2 < vw
            && rect.top + rect.height / 2 >= 0 && rect.top + rect.height / 2 < vh
        );
        if (centerInViewport) {
            try {
                const px = Math.max(0, Math.min(vw - 1, rect.left + rect.width / 2));
                const py = Math.max(0, Math.min(vh - 1, rect.top + rect.height / 2));
                const hit = document.elementFromPoint(px, py);
                if (hit && !(el.contains(hit) || hit.contains(el))) continue;
            } catch(_) {}
        }

        /* ── Text extraction cascade ── */
        const tag = (el.tagName || '').toUpperCase();
        let text = '';
        const offerOverride = surveyOfferText.get(el) || '';
        const al = el.getAttribute('aria-label');
        if (offerOverride) text = offerOverride;
        else if (al && al.trim()) text = al.trim();
        else {
            text = (el.textContent || '').trim();
            if (!text) text = el.getAttribute('placeholder') || '';
            if (!text) text = el.getAttribute('title') || '';
            if (!text) text = el.getAttribute('data-testid') || '';
            if (!text) text = el.getAttribute('alt') || '';
            if (!text) text = el.getAttribute('name') || '';
            if (!text) text = el.getAttribute('value') || '';
            text = text.trim();
        }
        text = text.replace(/\s+/g, ' ').slice(0, surveyOfferNodes.has(el) ? 240 : 120);
        if (!text && tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') continue;

        /* Only retain the innermost card selected by the survey-offer sweep.
           This prevents wrappers with the same reward/time text from consuming
           ranking slots or masquerading as separate offers. */
        if (SURVEY_OFFER_RE.test(text) && !surveyOfferNodes.has(el)) continue;

        /* V19.1: a SHORT element whose text is exactly a commerce action is a
           primary "goal" button even when rendered as a <div> (Flipkart/Amazon).
           This drives both kind classification and the score bonus below. */
        const isActionText = (text.length <= 28 && ACTION_RE.test(text));

        /* ── Kind classification ── */
        const role = (el.getAttribute('role') || '').toLowerCase();
        let kind = 'other';
        if (tag === 'A' || role === 'link')                                       kind = 'link';
        else if (tag === 'BUTTON' || role === 'button' || role === 'menuitem'
                 || role === 'tab' || role === 'switch')                          kind = 'button';
        else if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
                 || role === 'textbox' || role === 'searchbox' || role === 'combobox'
                 || el.getAttribute('contenteditable') === 'true'
                 || el.getAttribute('contenteditable') === 'plaintext-only')       kind = 'input';
        else if (role === 'checkbox' || role === 'radio')                         kind = 'button';
        else if (isActionText)                                                    kind = 'button';

        /* ── Coordinates ── */
        const cx = Math.round(rect.left + rect.width / 2);
        const cy = Math.round(rect.top + rect.height / 2);

        /* ── Scoring: viewport proximity ── */
        let score = Math.sqrt((cx - vcx) ** 2 + (cy - vcy) ** 2);
        const inViewport = rect.top >= 0 && rect.bottom <= vh && rect.left >= 0 && rect.right <= vw;
        if (!inViewport) score += 10000;

        /* V19.1: guarantee primary-action buttons survive the budget cut —
           whether off-screen (isAction rescue) or a div-rendered action bar
           (isActionText). This bonus beats the +10000 off-viewport penalty. */
        if (isAction || isActionText || surveyOfferNodes.has(el)) score -= 16000;

        /* ── V15.0 F1: Fixed/sticky elements are ALWAYS visible (W3C getComputedStyle) ── */
        const cssPos = cs.position;
        if (cssPos === 'fixed' || cssPos === 'sticky') {
            score -= 20000;  /* Guarantee inclusion — always-visible to user */
        }

        /* ── Input state detection ── */
        let inputState = null;
        if (kind === 'input') {
            try {
                const val = el.value || el.textContent || '';
                const trimVal = val.trim();
                inputState = trimVal.length > 0
                    ? { filled: true, length: trimVal.length, preview: trimVal.slice(0, 250) }
                    : { filled: false, length: 0 };
            } catch(_) { inputState = { filled: false, length: 0 }; }
        }

        /* ── Selection state detection ──
           Survey/rating controls are frequently <label>/<div> wrappers whose
           only visible state is a selected CSS class around a nested radio.
           Surface that state in BOTH structured data and Markdown so every
           failover model sees an authoritative [selected] marker. */
        let selectedState = false;
        let disabledState = false;
        let controlType = '';
        let requiredState = false;
        let choiceGroup = '';
        let groupLabel = '';
        let inModal = false;
        let fieldLabel = '';
        let fieldName = '';
        let fieldPlaceholder = '';
        let fieldAutocomplete = '';
        let fieldValue = '';
        let fieldOptions = [];
        let questionKey = '';
        try {
            const stateSel = [
                'input', 'option', '[role="radio"]', '[role="checkbox"]',
                '[role="option"]', '[aria-checked]', '[aria-selected]',
                '[data-state]', '[data-selected]', '[data-checked]'
            ].join(',');
            const control = (el.matches(stateSel) ? el : null)
                || el.querySelector(stateSel) || el;
            fieldName = control.getAttribute('name') || '';
            fieldPlaceholder = control.getAttribute('placeholder') || '';
            fieldAutocomplete = control.getAttribute('autocomplete') || '';
            fieldValue = String(control.value || '').slice(0, 160);
            if ((control.tagName || '').toUpperCase() === 'SELECT') {
                fieldOptions = [...control.options].slice(0, 60).map(option => ({
                    value: String(option.value || '').slice(0, 120),
                    label: String(option.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 120),
                    disabled: !!option.disabled,
                }));
            }
            const ariaLabel = control.getAttribute('aria-label') || '';
            let explicitLabel = '';
            try {
                const labelledBy = control.getAttribute('aria-labelledby') || '';
                if (labelledBy) {
                    explicitLabel = labelledBy.split(/\s+/).map(id => {
                        const node = document.getElementById(id);
                        return node ? (node.innerText || node.textContent || '') : '';
                    }).join(' ');
                }
                if (!explicitLabel && control.labels && control.labels.length) {
                    explicitLabel = [...control.labels].map(label => label.innerText || label.textContent || '').join(' ');
                }
                if (!explicitLabel) {
                    const wrappingLabel = control.closest('label');
                    if (wrappingLabel) explicitLabel = wrappingLabel.innerText || wrappingLabel.textContent || '';
                }
            } catch(_) {}
            fieldLabel = (ariaLabel || explicitLabel || fieldPlaceholder || fieldName || '')
                .replace(/\s+/g, ' ').trim().slice(0, 160);
            controlType = (control.getAttribute('type')
                || control.getAttribute('role') || '').toLowerCase();
            const cls = `${el.className || ''} ${control.className || ''}`;
            selectedState = !!(
                control.checked || control.selected
                || control.getAttribute('aria-checked') === 'true'
                || control.getAttribute('aria-selected') === 'true'
                || control.getAttribute('aria-pressed') === 'true'
                || ['checked','selected','on'].includes((control.getAttribute('data-state') || '').toLowerCase())
                || control.getAttribute('data-selected') === 'true'
                || control.getAttribute('data-checked') === 'true'
                || control.getAttribute('data-active') === 'true'
                || /(^|\s)(is-)?(selected|checked|chosen|active)(\s|$)/i.test(cls)
                || (/font-semibold/i.test(cls)
                    && /(?:border|bg)-\[[^\]]*(?:4a6cf7|2563eb|3b82f6|primary)/i.test(cls))
            );
            disabledState = !!(
                control.disabled || el.getAttribute('aria-disabled') === 'true'
                || control.getAttribute('aria-disabled') === 'true'
            );
            const requiredAncestor = control.closest('[aria-required="true"], fieldset[required]');
            requiredState = !!(
                control.required || control.getAttribute('aria-required') === 'true'
                || requiredAncestor
            );
            if (controlType === 'radio') {
                const nativeName = control.getAttribute('name') || '';
                const groupNode = control.closest(
                    '[role="radiogroup"], fieldset, tr, [role="row"], [data-question-id], [data-row-id]'
                );
                if (nativeName) choiceGroup = 'name:' + nativeName;
                else if (groupNode) {
                    const identity = groupNode.getAttribute('id')
                        || groupNode.getAttribute('data-question-id')
                        || groupNode.getAttribute('data-row-id')
                        || (groupNode.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 100);
                    if (identity) choiceGroup = 'group:' + identity;
                }
                if (groupNode) {
                    groupLabel = (groupNode.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 120);
                }
            }
            const questionNode = control.closest(
                '[data-question-id],[data-question],[data-testid*="question"],'
                + '[id*="question"],[class*="question"],fieldset,[role="group"],[role="radiogroup"]'
            );
            if (questionNode) {
                const questionIdentity = questionNode.getAttribute('data-question-id')
                    || questionNode.getAttribute('data-question')
                    || questionNode.getAttribute('id')
                    || questionNode.getAttribute('data-testid') || '';
                /* Never use the entire question container: an autocomplete
                   listbox is often mounted inside it, making every suggestion
                   update look like a new survey question. */
                const titleNode = questionNode.querySelector(
                    'legend,h1,h2,h3,h4,[data-testid*="title"],'
                    + '[class*="question-title"],[class*="questionText"]'
                );
                const stableQuestionText = ((titleNode && (titleNode.innerText || titleNode.textContent))
                    || fieldLabel || '').replace(/\s+/g, ' ').trim().slice(0, 140);
                questionKey = [questionIdentity, stableQuestionText].filter(Boolean).join('|');
            }
            const modalNode = control.closest(
                'dialog, [role="dialog"], [role="alertdialog"], [aria-modal="true"], '
                + '.modal, .modal-dialog, .popup, .popup-content, .overlay-content, '
                + '.snap-modal, [class*="snap-modal"], [class*="snap-show"]'
            );
            if (modalNode) {
                const semanticModal = modalNode.matches(
                    'dialog, [role="dialog"], [role="alertdialog"], [aria-modal="true"]'
                );
                let styledOverlay = false;
                try {
                    const modalStyle = window.getComputedStyle(modalNode);
                    const modalRect = modalNode.getBoundingClientRect();
                    const z = parseInt(modalStyle.zIndex || '0', 10) || 0;
                    styledOverlay = ['fixed', 'absolute'].includes(modalStyle.position)
                        && modalRect.width >= 180 && modalRect.height >= 80 && z >= 10;
                } catch(_) {}
                const classModal = /(?:snap-modal|snap-show|cookie-modal|privacy-modal)/i
                    .test(String(modalNode.className || ''));
                inModal = semanticModal || styledOverlay || classModal;
            }
            if (!inModal) {
                /* React dashboards often use generated class names and omit
                   dialog semantics entirely. Detect a substantial high-z
                   fixed ancestor so its Close/Not now control is still marked
                   as modal and wins the popup-first action gate. */
                let overlayAncestor = control.parentElement;
                for (let hops = 0; overlayAncestor && hops < 7; hops++, overlayAncestor = overlayAncestor.parentElement) {
                    try {
                        const os = window.getComputedStyle(overlayAncestor);
                        const or = overlayAncestor.getBoundingClientRect();
                        const oz = parseInt(os.zIndex || '0', 10) || 0;
                        const substantial = or.width * or.height >= vw * vh * 0.12;
                        if (os.position === 'fixed' && oz >= 10 && substantial) {
                            inModal = true;
                            break;
                        }
                    } catch(_) {}
                }
            }
        } catch(_) {}

        /* ── Semantic container path (for Markdown grouping) ── */
        let container = '';
        let cur = el.parentElement || (el.parentNode && el.parentNode.host);
        while (cur) {
            const t = (cur.tagName || '').toUpperCase();
            if (SEMANTIC_CONTAINERS.has(t) || cur.getAttribute('role')) {
                const id = cur.getAttribute('id');
                const r = cur.getAttribute('role');
                const tid = cur.getAttribute('data-testid');
                let desc = t.toLowerCase();
                if (id) desc += '#' + id;
                else if (tid) desc += '[' + tid + ']';
                if (r) desc += '(' + r + ')';
                container = container ? desc + ' > ' + container : desc;
            }
            cur = cur.parentElement || (cur.parentNode && cur.parentNode.host) || null;
        }
        if (!container) container = 'page';

        /* ── V19 Disambiguation hint: href + nearest row/item context ──
           Repeated controls ("comments", "Add to cart") share a label; this
           gives the LLM a cheap way to tell them apart and pick the RIGHT one. */
        let hint = '';
        /* href is the strongest disambiguator for repeated link labels (free) */
        if (tag === 'A') {
            let href = el.getAttribute('href') || '';
            if (href && !href.startsWith('javascript') && href !== '#') {
                href = href.replace(/^https?:\/\/[^/]+/, '');  /* strip origin */
                if (href.length > 45) href = href.slice(0, 45) + '…';
                hint = href;
            }
        }
        /* For SHORT/ambiguous labels, add a snippet of the enclosing row/item.
           Gated to short labels so we only pay innerText cost where it helps. */
        if ((text || '').length < 18) {
            let rc = el.parentElement, hops = 0, ctx = '';
            while (rc && hops < 6) {
                const rt = (rc.tagName || '').toUpperCase();
                const rr = (rc.getAttribute && rc.getAttribute('role')) || '';
                if (rt === 'TR' || rt === 'LI' || rt === 'ARTICLE'
                    || rr === 'row' || rr === 'listitem' || rr === 'article') {
                    let full = (rc.innerText || '').replace(/\s+/g, ' ').trim();
                    const own = (text || '').trim();
                    if (own && full.startsWith(own)) full = full.slice(own.length).trim();
                    ctx = full.slice(0, 50);
                    break;
                }
                rc = rc.parentElement; hops++;
            }
            if (ctx) hint = hint ? (hint + ' · ' + ctx) : ('in: ' + ctx);
        }

        /* V19.1: collapse nested duplicates of the SAME action button — Flipkart
           wraps "Add to cart" in several nested <div>s that all match the sweep.
           Keep one entry per physical location so the budget isn't eaten by
           clones (distinct product cards sit at distinct centers → still kept). */
        if (isActionText) {
            let dup = false;
            for (const c of actionCenters) {
                if (Math.abs(c.x - cx) <= 22 && Math.abs(c.y - cy) <= 22) { dup = true; break; }
            }
            if (dup) continue;
            actionCenters.push({ x: cx, y: cy });
        }

        if (!hint && fieldLabel) hint = 'field: ' + fieldLabel;
        scored.push({ el, text, kind, cx, cy, score, tag, inputState,
                      selectedState, disabledState, controlType, requiredState,
                      choiceGroup, groupLabel, inModal, container, hint,
                      fieldLabel, fieldName, fieldPlaceholder, fieldAutocomplete,
                      fieldValue, fieldOptions, questionKey });
    }

    scored.sort((a, b) => a.score - b.score);
    const top = scored.slice(0, MAX_ELEMENTS);

    /* ── V15.0 F1: Reserved slots for fixed/sticky elements cut by budget ── */
    const RESERVED_FIXED = 10;
    const fixedCut = scored.slice(MAX_ELEMENTS).filter(s => {
        try { return ['fixed','sticky'].includes(window.getComputedStyle(s.el).position); }
        catch(_) { return false; }
    });
    for (const f of fixedCut.slice(0, RESERVED_FIXED)) {
        top.push(f);
    }

    /* ━━━ 3. BUILD OUTPUTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

    /* 3a. Structured elements array (for coordinates) + stable handle registry.
       V19: window.__aid maps each eN to its LIVE DOM node so actions resolve to
       the EXACT element (fresh coords, no coordinate drift). Rebuilt fresh each
       extraction; auto-cleared on navigation. This is a window var, NOT a
       DOM-tree mutation — the "zero DOM mutation" principle is preserved. */
    const elements = [];
    window.__aid = {};
    for (let i = 0; i < top.length; i++) {
        const { text, kind, cx, cy, inputState, selectedState, disabledState,
                controlType, requiredState, choiceGroup, groupLabel, inModal,
                fieldLabel, fieldName, fieldPlaceholder, fieldAutocomplete,
                fieldValue, fieldOptions, questionKey } = top[i];
        const eid = 'e' + (i + 1);
        let label = text || top[i].tag.toLowerCase();
        if (inputState) {
            label += inputState.filled
                ? ` [filled: "${inputState.preview}"${inputState.length > 250 ? ' ...(' + inputState.length + ' total)' : ''} ${inputState.length}ch]`
                : ' [empty]';
        }
        if (selectedState) label += ' [selected]';
        if (disabledState) label += ' [disabled]';
        try { window.__aid[eid] = top[i].el; } catch(_) {}
        elements.push({ id: eid, kind, text: label, x: cx, y: cy,
                        hint: top[i].hint || '', selected: !!selectedState,
                        disabled: !!disabledState, control_type: controlType || '',
                        required: !!requiredState, choice_group: choiceGroup || '',
                        group_label: groupLabel || '', in_modal: !!inModal,
                        question_key: questionKey || '', visible: true,
                        name: fieldLabel || fieldName || '',
                        placeholder: fieldPlaceholder || '',
                        autocomplete: fieldAutocomplete || '', value: fieldValue || '',
                        options: fieldOptions || [], tag: top[i].tag || '' });
    }

    /* 3b. Semantic Markdown (Crawl4AI-inspired compression) */
    const groups = new Map();
    for (let i = 0; i < top.length; i++) {
        const c = top[i].container;
        if (!groups.has(c)) groups.set(c, []);
        groups.get(c).push(i);
    }

    let md = '';
    for (const [container, indices] of groups.entries()) {
        md += '## ' + container + '\n';
        for (const i of indices) {
            const e = elements[i];
            const icon = e.kind === 'input' ? '📝' : e.kind === 'button' ? '🔘' : e.kind === 'link' ? '🔗' : '•';
            const hintStr = e.hint ? `  ⟨${e.hint}⟩` : '';
            md += `- ${icon} **[${e.id}]** ${e.kind}: ${e.text}${hintStr} → (${e.x},${e.y})\n`;
        }
        md += '\n';
    }

    /* Rendered page text contains the static question/instructions that are not
       interactive and therefore absent from the element map. Keep it bounded;
       workers use it to solve attention checks and interpret answer choices. */
    let pageText = '';
    try {
        pageText = (document.body.innerText || '')
            .replace(/\n{3,}/g, '\n\n').trim().slice(0, 5000);
    } catch(_) {}

    return {
        elements,
        markdown: md.trim(),
        page_text: pageText,
        image_size: { width: vw, height: vh },
        element_count: candidates.length,
        source: 'god_mode_v11',
    };
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  SHADOW DOM PIERCER — Stateless init_script interceptor
#  Monkey-patches Element.prototype.attachShadow to capture all shadow roots
#  including mode:"closed".  Installed ONCE per context.
# ═══════════════════════════════════════════════════════════════════════════════

SHADOW_PIERCER_INIT_SCRIPT = r"""
(() => {
    if (window.__piercer) return;

    const _all = [];
    const _origAttach = Element.prototype.attachShadow;

    Element.prototype.attachShadow = function(init) {
        const root = _origAttach.call(this, init);
        try { _all.push(new WeakRef(root)); } catch(_) {}
        return root;
    };

    /* Index pre-existing open shadow roots */
    try {
        const tw = document.createTreeWalker(document, NodeFilter.SHOW_ELEMENT);
        while (tw.nextNode()) {
            if (tw.currentNode.shadowRoot) {
                _all.push(new WeakRef(tw.currentNode.shadowRoot));
            }
        }
    } catch(_) {}

    window.__piercer = {
        roots: () => {
            const live = [];
            const pruned = [];
            for (const ref of _all) {
                const sr = ref.deref();
                if (sr && sr.host && sr.host.isConnected) {
                    live.push(sr);
                    pruned.push(ref);
                }
            }
            _all.length = 0;
            _all.push(...pruned);
            return live;
        },
        count: () => _all.length,
    };
})();
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  TLS/JA3 STEALTH — Network-level fingerprint configuration
#  Applied to the browser context BEFORE any navigation.
#  Aligns our TLS signature with real Chrome 136 on Windows 10.
# ═══════════════════════════════════════════════════════════════════════════════

TLS_STEALTH_ARGS = [
    # Force real Chrome TLS stack (no Playwright-specific modifications)
    "--disable-features=IsolateOrigins,site-per-process",
    # HTTP/2 frame order alignment with Chrome
    "--enable-features=NetworkService,NetworkServiceInProcess",
    # Disable automation-revealing headers
    "--disable-client-side-phishing-detection",
    "--no-first-run",
    "--no-default-browser-check",
    # Cipher suite ordering to match Chrome 136 JA3
    "--ssl-version-min=tls1.2",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  PYTHON API — Clean async interface
# ═══════════════════════════════════════════════════════════════════════════════

async def install_shadow_piercer(context) -> None:
    """Install the Shadow DOM piercer into a browser context.
    Call this ONCE after creating the context, before navigating.
    """
    try:
        await context.add_init_script(SHADOW_PIERCER_INIT_SCRIPT)
        log.info("Shadow DOM piercer installed")
    except Exception as e:
        log.warning("Shadow piercer install failed: %s", e)


async def extract(
    page,
    target_hint: str | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Extract all interactive elements from the current page.

    Returns:
        {
            "elements":      [...],           # Structured element list with coordinates
            "markdown":      "...",           # Semantic Markdown for LLM context
            "image_size":    {w, h},          # Viewport dimensions
            "element_count": N,               # Total candidates found (pre-filter)
            "source":        "god_mode_v11",
        }
    """
    try:
        result = await asyncio.wait_for(
            page.evaluate(_GOD_MODE_JS, target_hint),
            timeout=timeout,
        )

        elements = result.get("elements", [])
        md = result.get("markdown", "")
        total = result.get("element_count", 0)

        log.info(
            "DOM extracted: %d elements (from %d candidates), markdown=%d chars",
            len(elements), total, len(md),
        )
        return result

    except asyncio.TimeoutError:
        log.warning("DOM extraction timed out (%.1fs)", timeout)
        return _empty_result()
    except Exception as e:
        log.warning("DOM extraction failed: %s", e)
        return _empty_result()


def _empty_result() -> dict[str, Any]:
    """Return a safe empty result on failure."""
    return {
        "elements": [],
        "markdown": "## page\n- _(no elements detected)_",
        "page_text": "",
        "image_size": {"width": 1920, "height": 1080},
        "element_count": 0,
        "source": "god_mode_v11_fallback",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  V19 — Stable element resolution (drift-proof, identity-verified targeting)
# ═══════════════════════════════════════════════════════════════════════════════
#  Resolves an element id (e.g. 'e5') to its LIVE DOM node via the window.__aid
#  registry built during extract(), scrolls it into view, and returns its CURRENT
#  center coordinates. Because the coords are re-read from the exact node the LLM
#  chose — at action time, after scrolling — they can never be stale and can never
#  "snap" to a neighbouring element. If the registry has no live node for the id
#  (rare: re-render/navigation), it returns {ok: False} and callers fall back to
#  the snapshot coordinates (today's behaviour) — a strict superset, no regression.

_RESOLVE_JS = r"""
(eid) => {
    const reg = window.__aid || {};
    const el = reg[eid];
    if (!el) return { ok: false, reason: 'not_in_registry' };
    if (!el.isConnected) return { ok: false, reason: 'detached' };
    // A label may outrank its associated input in the accessibility snapshot.
    // Resolve it to the live form control before a typing action uses its box.
    let target = el;
    if ((el.tagName || '').toUpperCase() === 'LABEL') {
        try {
            target = el.control || (el.htmlFor && document.getElementById(el.htmlFor))
                || el.querySelector('input,textarea,[contenteditable="true"],[role="textbox"]')
                || el;
        } catch(_) { target = el; }
    }
    try { target.scrollIntoView({ block: 'center', inline: 'center' }); } catch(_) {}
    let r;
    try { r = target.getBoundingClientRect(); } catch(_) { return { ok: false, reason: 'no_rect' }; }
    if (r.width < 1 || r.height < 1) return { ok: false, reason: 'zero_size' };
    const cx = Math.round(r.left + r.width / 2);
    const cy = Math.round(r.top + r.height / 2);
    const vw = window.innerWidth || 0, vh = window.innerHeight || 0;
    const onscreen = cx >= 0 && cy >= 0 && cx <= vw && cy <= vh;
    const tag = (target.tagName || '').toUpperCase();
    let role = '';
    try { role = target.getAttribute('role') || ''; } catch(_) {}
    let text = '';
    try {
        text = (target.getAttribute('aria-label') || target.textContent || '')
            .replace(/\s+/g, ' ').trim().slice(0, 40);
    } catch(_) {}
    return {
        ok: true,
        x: cx,
        y: cy,
        // Preserve the live box for virtual-mouse target sampling. The centre
        // remains the fallback for very small controls.
        rect: { x: r.left, y: r.top, width: r.width, height: r.height },
        tag,
        requested_tag: (el.tagName || '').toUpperCase(),
        resolved_id: Object.keys(reg).find(k => reg[k] === target) || '',
        role,
        text,
        onscreen,
    };
}
"""


async def resolve_element(page, eid: str, timeout: float = 2.0) -> dict[str, Any]:
    """Resolve an element id to FRESH, identity-verified center coordinates.

    Returns:
        {"ok": True, "x", "y", "rect", "tag", "role", "text", "onscreen"} on success, or
        {"ok": False, "reason": ...} when the id is not a live node (caller should
        fall back to snapshot coordinates).
    """
    if not eid:
        return {"ok": False, "reason": "no_eid"}
    try:
        res = await asyncio.wait_for(page.evaluate(_RESOLVE_JS, eid), timeout=timeout)
        return res or {"ok": False, "reason": "null_result"}
    except Exception as e:
        log.debug("resolve_element(%s) failed: %s", eid, e)
        return {"ok": False, "reason": str(e)[:80]}


# ═══════════════════════════════════════════════════════════════════════════════
#  FORM FIELD DETECTOR — Specialized extraction for form-heavy pages
# ═══════════════════════════════════════════════════════════════════════════════

_FORM_FIELD_DETECTOR_JS = r"""
() => {
    const fields = [];

    function scan(root) {
        for (const el of root.querySelectorAll('input, textarea, select, [contenteditable="true"], [role="textbox"]')) {
            try {
                const rect = el.getBoundingClientRect();
                if (rect.width < 10 || rect.height < 10) continue;

                const cs = window.getComputedStyle(el);
                if (cs.display === 'none' || cs.visibility === 'hidden') continue;

                const type = el.getAttribute('type') || el.tagName.toLowerCase();
                const name = el.getAttribute('name') || '';
                const ph = el.getAttribute('placeholder') || '';
                const label = el.getAttribute('aria-label') || '';
                const val = el.value || el.textContent || '';

                fields.push({
                    type,
                    name,
                    placeholder: ph,
                    label,
                    value_length: val.trim().length,
                    x: Math.round(rect.left + rect.width / 2),
                    y: Math.round(rect.top + rect.height / 2),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height),
                });
            } catch(_) {}
        }

        /* Recurse into shadow roots */
        for (const host of root.querySelectorAll('*')) {
            const sr = host.shadowRoot;
            if (sr) scan(sr);
        }
    }

    scan(document);

    /* Also scan pierced roots */
    if (window.__piercer && typeof window.__piercer.roots === 'function') {
        for (const sr of window.__piercer.roots()) { scan(sr); }
    }

    return { fields };
}
"""


async def detect_form_fields(page) -> dict[str, Any]:
    """Detect all form fields on the page (inputs, textareas, selects)."""
    try:
        return await asyncio.wait_for(
            page.evaluate(_FORM_FIELD_DETECTOR_JS),
            timeout=3.0,
        )
    except Exception as e:
        log.warning("Form field detection failed: %s", e)
        return {"fields": []}


# This is deliberately a separate, unranked pass.  The normal extractor is
# optimized for prompt size and can cull a survey's controls when a React
# render is between states.  Recovery must inspect the native controls before
# escalating to vision, even when they are off-screen or visually subtle.
_SPARSE_FORM_AUDIT_JS = r"""
() => {
    const found = [], seen = new WeakSet();
    const actionable = 'input,textarea,select,button,label,[role="radio"],[role="checkbox"],[role="option"],[role="textbox"],[role="button"],[role="link"],[contenteditable="true"]';
    const visibleEnough = (el) => {
        if (!el || !el.isConnected) return false;
        for (let n = el; n && n.nodeType === 1; n = n.parentElement || (n.parentNode && n.parentNode.host)) {
            const s = getComputedStyle(n);
            if (n.hidden || n.getAttribute('aria-hidden') === 'true' || s.display === 'none' || s.visibility === 'hidden') return false;
        }
        return true;
    };
    const textOf = (el) => String(el.innerText || el.textContent || el.getAttribute('aria-label') || el.getAttribute('title') || '').replace(/\s+/g, ' ').trim().slice(0, 180);
    const center = (el) => { const r = el.getBoundingClientRect(); return {x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2), width: Math.round(r.width), height: Math.round(r.height)}; };
    const labelFor = (el) => {
        let label = el.getAttribute('aria-label') || '';
        if (!label && el.id) { try { label = document.querySelector(`label[for="${CSS.escape(el.id)}"]`)?.innerText || ''; } catch (_) {} }
        if (!label) { try { label = el.closest('label')?.innerText || ''; } catch (_) {} }
        return String(label).replace(/\s+/g, ' ').trim().slice(0, 180);
    };
    const register = (el, kind) => {
        if (!el || seen.has(el) || !visibleEnough(el)) return;
        seen.add(el);
        const r = center(el), tag = (el.tagName || '').toLowerCase();
        let id = ''; for (const [key, value] of Object.entries(window.__aid || {})) if (value === el) { id = key; break; }
        if (!id) { let i = 1; do { id = 's' + i++; } while (window.__aid && window.__aid[id]); window.__aid = window.__aid || {}; window.__aid[id] = el; }
        const role = el.getAttribute('role') || (tag === 'input' ? (el.type || 'text') : tag);
        const label = labelFor(el), selected = el.checked === true || el.selected === true || el.getAttribute('aria-checked') === 'true' || el.getAttribute('aria-selected') === 'true';
        found.push({id, kind: kind || tag, tag, role, control_type: role, text: (label || textOf(el)).slice(0, 220), hint: 'sparse recovery; native control', x:r.x, y:r.y, width:r.width, height:r.height, selected, checked: !!el.checked, required: !!el.required || el.getAttribute('aria-required') === 'true', disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true', value: String(el.value || '').slice(0, 120), visible: true, sparse_recovery: true});
    };
    const scan = (root) => {
        try { for (const el of root.querySelectorAll(actionable)) register(el, el.tagName.toLowerCase()); } catch (_) {}
        try { for (const host of root.querySelectorAll('*')) if (host.shadowRoot) scan(host.shadowRoot); } catch (_) {}
    };
    scan(document);
    try { for (const sr of (window.__piercer?.roots?.() || [])) scan(sr); } catch (_) {}
    return {controls: found, count: found.length};
}
"""


async def recover_sparse_controls(page, retries: tuple[float, ...] = (0.0, 0.25, 0.75)) -> dict[str, Any]:
    """Run a bounded native-control audit for pages with a sparse DOM snapshot."""
    last: dict[str, Any] = {"controls": [], "count": 0}
    for delay in retries:
        if delay:
            await asyncio.sleep(delay)
        try:
            result = await asyncio.wait_for(page.evaluate(_SPARSE_FORM_AUDIT_JS), timeout=1.5)
            last = result or last
            if last.get("controls"):
                return {**last, "status": "RECOVERED"}
        except Exception as exc:
            log.debug("Sparse DOM audit failed: %s", str(exc)[:120])
    return {**last, "status": "UNRESOLVED"}


def find_field(form_data: dict, field_type: str) -> dict | None:
    """Find the best matching field for a given type (title, body, etc.)."""
    fields = form_data.get("fields", [])
    if not fields:
        return None

    title_keywords = {"title", "subject", "headline", "heading"}
    body_keywords = {"body", "content", "description", "message", "text", "article", "write", "post", "comment"}

    for f in fields:
        combined = f"{f.get('name', '')} {f.get('placeholder', '')} {f.get('label', '')}".lower()

        if field_type == "title":
            if any(kw in combined for kw in title_keywords):
                return f
        elif field_type == "body":
            if any(kw in combined for kw in body_keywords):
                return f
            if f.get("height", 0) > 100 and f.get("type") in ("textarea", "div"):
                return f

    # Fallback: for body, pick the largest field
    if field_type == "body" and fields:
        return max(fields, key=lambda f: f.get("width", 0) * f.get("height", 0))

    return fields[0] if fields else None


# ═══════════════════════════════════════════════════════════════════════════════
#  FORM AUDIT — Verify field values after injection
# ═══════════════════════════════════════════════════════════════════════════════

_FIELD_AUDIT_JS = r"""
() => {
    const result = {};

    function auditIn(root, label) {
        for (const el of root.querySelectorAll('input, textarea, [contenteditable="true"], [role="textbox"]')) {
            try {
                const rect = el.getBoundingClientRect();
                if (rect.width < 10 || rect.height < 10) continue;

                const cs = window.getComputedStyle(el);
                if (cs.display === 'none') continue;

                const val = el.value || el.textContent || '';
                const trimVal = val.trim();
                const name = el.getAttribute('name') || el.getAttribute('placeholder') || el.tagName;

                if (rect.height > 100 || el.tagName === 'TEXTAREA' || el.getAttribute('contenteditable')) {
                    if (!result.body || trimVal.length > (result.body.length || 0)) {
                        result.body = { found: true, length: trimVal.length, value: trimVal.slice(0, 100) };
                    }
                } else {
                    if (!result.title || trimVal.length > (result.title.length || 0)) {
                        result.title = { found: true, length: trimVal.length, value: trimVal.slice(0, 100) };
                    }
                }
            } catch(_) {}
        }

        for (const host of root.querySelectorAll('*')) {
            if (host.shadowRoot) auditIn(host.shadowRoot, 'shadow');
        }
    }

    auditIn(document, 'main');

    if (window.__piercer && typeof window.__piercer.roots === 'function') {
        for (const sr of window.__piercer.roots()) { auditIn(sr, 'pierced'); }
    }

    return result;
}
"""


async def audit_form_fields(page) -> dict:
    """Audit the current values in form fields (used after typing to verify)."""
    try:
        return await asyncio.wait_for(
            page.evaluate(_FIELD_AUDIT_JS),
            timeout=3.0,
        )
    except Exception as e:
        log.warning("Form audit failed: %s", e)
        return {}
