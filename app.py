
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Research Agent",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from config import settings
from memory.vector_store import ResearchVectorStore
from models import Paper, ResearchSession
from orchestrator import run_research_session


# ── Global CSS — minimalist, monospace-accent aesthetic ───────────────────────
# We inject one <style> block at startup. All component styling lives here
# rather than being scattered across st.markdown() calls.
STYLE = """
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root tokens ── */
:root {
    --fg:        #0f0f0f;
    --fg-muted:  #6b6b6b;
    --fg-faint:  #aaaaaa;
    --bg:        #fafafa;
    --bg-card:   #ffffff;
    --bg-hover:  #f3f3f3;
    --border:    #e8e8e8;
    --accent:    #1a1a1a;
    --accent2:   #4f46e5;   /* indigo — used sparingly */
    --radius:    8px;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'Inter', sans-serif;
}

/* ── Global resets ── */
html, body, [class*="css"] { font-family: var(--sans); color: var(--fg); }
.block-container { max-width: 860px; padding: 2rem 1.5rem 4rem; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Typography ── */
h1 { font-size: 1.5rem; font-weight: 500; letter-spacing: -0.5px; margin-bottom: 0.15rem; }
h2 { font-size: 1rem; font-weight: 500; color: var(--fg-muted); margin: 0; }
h3 { font-size: 0.8rem; font-weight: 500; text-transform: uppercase;
     letter-spacing: 0.08em; color: var(--fg-faint); margin: 0 0 0.75rem; }

/* ── Search input ── */
.stTextInput > div > div > input {
    font-family: var(--mono);
    font-size: 0.95rem;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem 1rem !important;
    background: var(--bg-card) !important;
    color: var(--fg) !important;
    box-shadow: none !important;
    transition: border-color 0.15s;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: none !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.4rem !important;
    letter-spacing: 0.02em;
    transition: opacity 0.15s;
}
.stButton > button[kind="primary"]:hover { opacity: 0.82; }

/* ── Secondary button ── */
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--fg-muted) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.2rem !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--fg-muted) !important;
    color: var(--fg) !important;
}

/* ── Paper card ── */
.paper-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.15s, background 0.15s;
}
.paper-card:hover { border-color: #c0c0c0; background: var(--bg-hover); }
.paper-title {
    font-size: 0.92rem;
    font-weight: 500;
    color: var(--fg);
    line-height: 1.45;
    margin-bottom: 0.25rem;
}
.paper-meta {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--fg-faint);
    margin-bottom: 0.5rem;
}
.paper-abstract {
    font-size: 0.82rem;
    color: var(--fg-muted);
    line-height: 1.6;
}
.score-pill {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--fg-faint);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 0.1rem 0.55rem;
    margin-left: 0.4rem;
    vertical-align: middle;
}
.source-tag {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 500;
    background: var(--bg);
    color: var(--fg-muted);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.1rem 0.45rem;
    margin-left: 0.3rem;
    vertical-align: middle;
}

/* ── Section divider ── */
.section-label {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--fg-faint);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    margin: 1.5rem 0 1rem;
}

/* ── Interest map bar ── */
.imap-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
}
.imap-topic { flex: 0 0 220px; color: var(--fg); white-space: nowrap;
              overflow: hidden; text-overflow: ellipsis; }
.imap-bar-wrap { flex: 1; background: var(--border); border-radius: 99px; height: 4px; }
.imap-bar { height: 4px; border-radius: 99px; background: var(--accent); }
.imap-weight { flex: 0 0 38px; text-align: right; font-family: var(--mono);
               font-size: 0.7rem; color: var(--fg-faint); }

/* ── Feedback textarea ── */
.stTextArea textarea {
    font-family: var(--sans) !important;
    font-size: 0.88rem !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-card) !important;
    color: var(--fg) !important;
    box-shadow: none !important;
    resize: vertical;
    transition: border-color 0.15s;
}
.stTextArea textarea:focus { border-color: var(--accent) !important; }

/* ── Spinner / status ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Toast / alert overrides ── */
.stAlert { border-radius: var(--radius) !important; font-size: 0.85rem; }

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--border);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem;
    font-weight: 400;
    color: var(--fg-muted);
    padding: 0.5rem 1rem;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--fg) !important;
    border-bottom: 2px solid var(--accent) !important;
    font-weight: 500;
}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)


# ── Session-state initialisation ──────────────────────────────────────────────
# WHY session_state?  Streamlit re-runs the entire script on every interaction.
# session_state persists values (like the vector store and past results)
# across re-runs within the same browser session.

def _init_state() -> None:
    if "vector_store" not in st.session_state:
        with st.spinner("Loading memory store…"):
            st.session_state.vector_store = ResearchVectorStore()
    if "session_history" not in st.session_state:
        st.session_state.session_history: list[ResearchSession] = []
    if "current_session" not in st.session_state:
        st.session_state.current_session: ResearchSession | None = None
    if "running" not in st.session_state:
        st.session_state.running = False

_init_state()


# ── Helper: render a single paper card ───────────────────────────────────────

def _paper_card(paper: Paper, show_abstract: bool = True) -> None:
    score_html = (
        f'<span class="score-pill">{paper.relevance_score:.2f}</span>'
        if paper.relevance_score > 0 else ""
    )
    source_html = f'<span class="source-tag">{paper.source}</span>'
    authors = ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else "")
    abstract_html = (
        f'<div class="paper-abstract">{paper.abstract[:280]}{"…" if len(paper.abstract) > 280 else ""}</div>'
        if show_abstract and paper.abstract and paper.abstract != "(Seen in previous session — abstract not stored)"
        else ""
    )

    url_attr = f'href="{paper.url}" target="_blank"' if paper.url else ""
    title_html = (
        f'<a {url_attr} style="text-decoration:none; color:inherit;">{paper.title}</a>'
        if paper.url else paper.title
    )

    st.markdown(f"""
    <div class="paper-card">
        <div class="paper-title">{title_html}{score_html}{source_html}</div>
        <div class="paper-meta">{authors} &nbsp;·&nbsp; {paper.published_date or "—"}</div>
        {abstract_html}
    </div>
    """, unsafe_allow_html=True)


# ── Helper: interest map bars ─────────────────────────────────────────────────

def _render_interest_map(interest_map: dict[str, float]) -> None:
    if not interest_map:
        st.markdown('<span style="color:var(--fg-faint);font-size:0.85rem;">No history yet — run a few searches to build your map.</span>', unsafe_allow_html=True)
        return
    top = list(interest_map.items())[:12]
    for topic, weight in top:
        bar_pct = int(weight * 100)
        st.markdown(f"""
        <div class="imap-row">
            <div class="imap-topic" title="{topic}">{topic}</div>
            <div class="imap-bar-wrap"><div class="imap-bar" style="width:{bar_pct}%"></div></div>
            <div class="imap-weight">{weight:.3f}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Email feedback ────────────────────────────────────────────────────────────

def _send_feedback_email(name: str, feedback: str, rating: str) -> bool:
    """
    Send feedback via Gmail SMTP using an App Password.

    Setup steps:
      1. Enable 2FA on your Google account.
      2. Go to myaccount.google.com → Security → App Passwords.
      3. Generate a password for "Mail / Other".
      4. Add FEEDBACK_EMAIL_FROM and FEEDBACK_EMAIL_APP_PASSWORD to .env.

    WHY Gmail SMTP over a third-party service?
      Zero external dependencies — no Formspree account, no API key, no
      webhook. Pure Python stdlib (smtplib + email). The downside is it
      requires an App Password and Google account, which is a fair trade
      for a personal research tool.
    """
    sender = settings.feedback_email_from
    recipient = settings.feedback_email_to
    password = settings.feedback_email_app_password

    if not sender or not password:
        return False  # Caller shows a graceful message

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Research Agent] Feedback — {rating} stars"
    msg["From"] = sender
    msg["To"] = recipient

    body_text = f"Name: {name}\nRating: {rating}/5\n\nFeedback:\n{feedback}"
    body_html = f"""
    <html><body style="font-family:Inter,sans-serif;color:#0f0f0f;max-width:600px">
    <h2 style="font-weight:500;font-size:1.1rem">New Feedback — Research Agent</h2>
    <table style="border-collapse:collapse;width:100%">
      <tr><td style="padding:6px 0;color:#6b6b6b;width:80px">Name</td><td>{name or "Anonymous"}</td></tr>
      <tr><td style="padding:6px 0;color:#6b6b6b">Rating</td><td>{"★" * int(rating)}{"☆" * (5 - int(rating))} ({rating}/5)</td></tr>
    </table>
    <hr style="border:none;border-top:1px solid #e8e8e8;margin:1rem 0">
    <p style="line-height:1.6">{feedback.replace(chr(10), "<br>")}</p>
    </body></html>
    """
    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        return True
    except Exception:
        return False


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom:2rem">
  <h1>⬡ Research Agent</h1>
  <h2>Multi-agent academic paper discovery</h2>
</div>
""", unsafe_allow_html=True)

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_search, tab_history, tab_feedback = st.tabs(["Search", "Interest map", "Feedback"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SEARCH
# ══════════════════════════════════════════════════════════════════════════════

with tab_search:
    # Query input row
    col_input, col_btn = st.columns([5, 1], gap="small")
    with col_input:
        query = st.text_input(
            label="query",
            placeholder="e.g. RL-based machine unlearning",
            label_visibility="collapsed",
        )
    with col_btn:
        # Extra top-margin so button aligns with input
        st.markdown("<div style='margin-top:4px'>", unsafe_allow_html=True)
        search_clicked = st.button("Search", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Hint text
    st.markdown(
        '<span style="font-size:0.75rem;color:var(--fg-faint);">'
        'Agent 1 searches your exact topic · Agent 2 broadens to related fields · '
        'Agent 3 connects to your research history</span>',
        unsafe_allow_html=True,
    )

    # ── Run agents ────────────────────────────────────────────────────────────
    if search_clicked and query.strip():
        st.session_state.running = True
        progress_area = st.empty()

        with progress_area.container():
            # Three-step status strip
            s1, s2, s3 = st.columns(3)
            p1 = s1.empty()
            p2 = s2.empty()
            p3 = s3.empty()

            def _status(col_ph, icon, label, state):
                # state: "wait" | "run" | "done" | "err"
                colours = {"wait": "#aaa", "run": "#1a1a1a", "done": "#16a34a", "err": "#dc2626"}
                col_ph.markdown(
                    f'<div style="font-size:0.8rem;color:{colours[state]};padding:0.3rem 0">'
                    f'{icon} {label}</div>',
                    unsafe_allow_html=True,
                )

            _status(p1, "○", "Focused search", "run")
            _status(p2, "○", "Broader context", "wait")
            _status(p3, "○", "Interest map", "wait")

        try:
            # We call orchestrator directly — it already sequences the agents.
            # For a live progress feel we patch a lightweight status bar.
            t0 = time.time()

            # Import agent runners for granular status updates
            from agents.focused_search_agent import run_focused_search_agent
            from agents.broader_context_agent import run_broader_context_agent
            from agents.interest_map_agent import run_interest_map_agent
            from models import ResearchSession as RS

            session = RS(query=query.strip())

            # Agent 1
            a1 = run_focused_search_agent(query.strip())
            session.focused_papers = a1.papers
            _status(p1, "●", f"Focused search · {len(a1.papers)} papers", "done")
            _status(p2, "○", "Broader context", "run")

            # Agent 2
            a1_ids = {p.paper_id for p in a1.papers}
            a2 = run_broader_context_agent(query.strip(), exclude_paper_ids=a1_ids)
            session.broader_papers = a2.papers
            _status(p2, "●", f"Broader context · {len(a2.papers)} papers", "done")
            _status(p3, "○", "Interest map", "run")

            # Agent 3
            all_papers = a1.papers + a2.papers
            a3 = run_interest_map_agent(
                query.strip(),
                current_session_papers=all_papers,
                vector_store=st.session_state.vector_store,
            )
            session.interest_papers = a3.papers
            session.interest_map_summary = a3.reasoning
            _status(p3, "●", f"Interest map · {len(a3.papers)} papers", "done")

            st.session_state.current_session = session
            st.session_state.session_history.append(session)
            elapsed = time.time() - t0

            progress_area.markdown(
                f'<div style="font-size:0.75rem;color:var(--fg-faint);margin-top:0.25rem">'
                f'Completed in {elapsed:.1f} s</div>',
                unsafe_allow_html=True,
            )

        except Exception as exc:
            progress_area.error(f"Search failed: {exc}")
            st.stop()

        st.session_state.running = False

    # ── Results display ───────────────────────────────────────────────────────
    session: ResearchSession | None = st.session_state.current_session

    if session:
        # ── Agent 1 ───────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">🎯 &nbsp;Focused — exact topic</div>', unsafe_allow_html=True)
        if session.focused_papers:
            for p in session.focused_papers:
                _paper_card(p)
        else:
            st.markdown('<span style="color:var(--fg-faint);font-size:0.85rem">No focused papers found.</span>', unsafe_allow_html=True)

        # ── Agent 2 ───────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">🌐 &nbsp;Broader context — related fields</div>', unsafe_allow_html=True)
        if session.broader_papers:
            for p in session.broader_papers:
                _paper_card(p)
        else:
            st.markdown('<span style="color:var(--fg-faint);font-size:0.85rem">No broader papers found.</span>', unsafe_allow_html=True)

        # ── Agent 3 ───────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">🧠 &nbsp;From your history — connected research</div>', unsafe_allow_html=True)
        if session.interest_papers:
            for p in session.interest_papers:
                _paper_card(p, show_abstract=False)
        else:
            st.markdown('<span style="color:var(--fg-faint);font-size:0.85rem">No historical papers yet — run more searches to build your interest map.</span>', unsafe_allow_html=True)

        # ── Interest narrative ─────────────────────────────────────────────────
        if session.interest_map_summary:
            st.markdown('<div class="section-label">📍 &nbsp;Your research profile</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:0.88rem;color:var(--fg-muted);line-height:1.7;'
                f'padding:1rem 1.25rem;border:1px solid var(--border);border-radius:var(--radius);'
                f'background:var(--bg-card)">{session.interest_map_summary}</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INTEREST MAP
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    interest_map = st.session_state.vector_store.build_interest_map()
    total_searches = st.session_state.vector_store._search_col.count()

    # Stats row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div style="font-family:var(--mono);font-size:1.6rem;font-weight:500">{total_searches}</div>'
            f'<div style="font-size:0.75rem;color:var(--fg-faint)">sessions recorded</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div style="font-family:var(--mono);font-size:1.6rem;font-weight:500">{len(interest_map)}</div>'
            f'<div style="font-size:0.75rem;color:var(--fg-faint)">unique topics</div>',
            unsafe_allow_html=True,
        )
    with c3:
        papers_count = st.session_state.vector_store._papers_col.count()
        st.markdown(
            f'<div style="font-family:var(--mono);font-size:1.6rem;font-weight:500">{papers_count}</div>'
            f'<div style="font-size:0.75rem;color:var(--fg-faint)">papers stored</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">Interest map &nbsp;— weighted by recency</div>', unsafe_allow_html=True)
    _render_interest_map(interest_map)

    # Decay explainer
    st.markdown(
        '<div style="margin-top:1.25rem;font-size:0.75rem;color:var(--fg-faint)">'
        'Bar width = cosine similarity × exp(−0.05 × days_ago). '
        'Searches older than ~60 days contribute &lt;5% weight.</div>',
        unsafe_allow_html=True,
    )

    # Clear history
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    if st.button("Clear history", type="secondary"):
        try:
            vs = st.session_state.vector_store
            vs._client.delete_collection("search_history")
            vs._client.delete_collection("papers_seen")
            del st.session_state["vector_store"]
            st.session_state.session_history = []
            st.session_state.current_session = None
            st.rerun()
        except Exception as exc:
            st.error(f"Could not clear history: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

with tab_feedback:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.88rem;color:var(--fg-muted);margin-bottom:1.25rem;line-height:1.6">'
        'Suggestions, bug reports, or feature requests — all welcome. '
        'This goes directly to the maintainer\'s inbox.</div>',
        unsafe_allow_html=True,
    )

    fb_name = st.text_input("Name (optional)", placeholder="Anonymous", label_visibility="visible")
    fb_rating = st.select_slider(
        "How useful was this session?",
        options=["1", "2", "3", "4", "5"],
        value="3",
    )
    fb_text = st.text_area(
        "Your feedback",
        placeholder="What worked well? What's missing? Bugs? Ideas?",
        height=140,
    )

    email_configured = bool(settings.feedback_email_from and settings.feedback_email_app_password)

    col_send, col_note = st.columns([2, 5], gap="small")
    with col_send:
        send_clicked = st.button("Send feedback", type="primary", use_container_width=True)
    with col_note:
        if not email_configured:
            st.markdown(
                '<div style="font-size:0.75rem;color:var(--fg-faint);padding-top:0.7rem">'
                'Email not configured — set FEEDBACK_EMAIL_FROM and '
                'FEEDBACK_EMAIL_APP_PASSWORD in .env</div>',
                unsafe_allow_html=True,
            )

    if send_clicked:
        if not fb_text.strip():
            st.warning("Please write something before sending.")
        elif not email_configured:
            st.info("Email not configured. Add FEEDBACK_EMAIL_FROM and FEEDBACK_EMAIL_APP_PASSWORD to your .env file.")
        else:
            with st.spinner("Sending…"):
                ok = _send_feedback_email(fb_name, fb_text, fb_rating)
            if ok:
                st.success("Sent — thank you!")
            else:
                st.error("Failed to send. Check your SMTP settings in .env.")
