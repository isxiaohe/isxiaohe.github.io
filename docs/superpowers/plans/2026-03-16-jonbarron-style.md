# Jon Barron Style Website Redesign Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Hugo academic website to match Jon Barron's academic style with Lato font, 800px max-width, publication thumbnails with hover effects, and unified link bar.

**Architecture:** Single-page homepage with bio, photo placeholder, and publications. Sub-pages (Notes, Paper Reading) accessible via links in bio section. No top navigation bar.

**Tech Stack:** Hugo static site generator, HTML templates, CSS with hover effects

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `hugo.toml` | Modify | Add scholar URL parameter |
| `static/css/style.css` | Rewrite | Lato font, colors, hover effects, publication cards |
| `layouts/_default/baseof.html` | Rewrite | Remove nav, add Lato font |
| `layouts/index.html` | Rewrite | Jon Barron style homepage |
| `layouts/notes/list.html` | Modify | Add intro text |
| `content/publications/g4splat.md` | Modify | Update author symbols |

---

## Chunk 1: CSS Foundation

### Task 1: Rewrite CSS with Jon Barron Style

**Files:**
- Rewrite: `static/css/style.css`

- [ ] **Step 1: Write the complete CSS file**

```css
/* Lato Font from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,400;0,700;1,400;1,700&display=swap');

/* Root variables */
:root {
    --max-width: 800px;
    --link-blue: #1772d0;
    --link-hover: #f09228;
    --highlight-bg: #ffffd0;
    --text-color: #222222;
    --text-muted: #666666;
}

/* Base styles */
body {
    font-family: 'Lato', Verdana, Helvetica, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-color);
    background-color: #ffffff;
    margin: 0;
    padding: 0;
}

/* Typography */
td, th, tr, p, a, strong {
    font-family: 'Lato', Verdana, Helvetica, sans-serif;
    font-size: 14px;
}

h2 {
    margin: 0;
    font-weight: normal;
    font-size: 22px;
}

/* Links */
a {
    color: var(--link-blue);
    text-decoration: none;
}

a:focus,
a:hover {
    color: var(--link-hover);
    text-decoration: none;
}

/* Name title */
.name {
    padding-top: 20px;
    margin: 0;
    font-size: 32px;
    font-weight: normal;
}

/* Paper title */
.papertitle {
    font-weight: 700;
}

/* Container - centered 800px */
.container {
    width: 100%;
    max-width: var(--max-width);
    margin: 0 auto;
    padding: 0;
}

/* Profile photo placeholder */
.profile-photo {
    width: 160px;
    height: 160px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 48px;
    font-weight: bold;
}

/* Publication thumbnail hover effects */
.one {
    width: 160px;
    height: 160px;
    position: relative;
}

.two {
    width: 160px;
    height: 160px;
    position: absolute;
    transition: opacity .2s ease-in-out;
    -moz-transition: opacity .2s ease-in-out;
    -webkit-transition: opacity .2s ease-in-out;
}

.fade {
    transition: opacity .2s ease-in-out;
    -moz-transition: opacity .2s ease-in-out;
    -webkit-transition: opacity .2s ease-in-out;
}

/* Highlight for selected papers */
span.highlight {
    background-color: var(--highlight-bg);
}

/* Notes page styles */
.notes-intro {
    color: var(--text-muted);
    margin-bottom: 30px;
}

.note-item-container {
    border-left: 3px solid #eeeeee;
    padding-left: 20px;
    margin-bottom: 30px;
    transition: border-color 0.2s;
}

.note-item-container:hover {
    border-left-color: var(--link-blue);
}

.note-item-title {
    margin: 0;
    font-size: 1.1rem;
    line-height: 1.3;
}

.note-item-title a {
    text-decoration: none;
    color: var(--text-color);
    font-weight: 600;
}

.note-item-title a:hover {
    color: var(--link-blue);
}

.note-item-meta {
    font-size: 0.9em;
    color: var(--text-muted);
    margin-top: 4px;
}

.note-item-summary {
    font-size: 0.88em;
    margin-top: 8px;
    color: #444444;
    line-height: 1.5;
}

/* Footer */
footer {
    margin-top: 60px;
    padding: 20px 0;
    font-size: 0.85em;
    color: var(--text-muted);
    text-align: center;
}
```

- [ ] **Step 2: Verify CSS file is written correctly**

Run: `head -20 /Users/matthew-xh/Study/CS/my-academic-site/static/css/style.css`
Expected: CSS content starting with `@import url('https://fonts.googleapis.com...`

- [ ] **Step 3: Commit CSS changes**

```bash
git add static/css/style.css
git commit -m "style: rewrite CSS with Jon Barron style

- Add Lato font from Google Fonts
- 800px max-width, centered layout
- Blue (#1772d0) to orange (#f09228) link hover
- Publication thumbnail hover effects
- Notes page styles

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 2: Base Template

### Task 2: Update Base Template

**Files:**
- Rewrite: `layouts/_default/baseof.html`

- [ ] **Step 1: Write the new base template**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ if .IsHome }}{{ .Site.Title }}{{ else }}{{ .Title }} | {{ .Site.Title }}{{ end }}</title>

    <!-- Lato Font -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,400;0,700;1,400;1,700&display=swap" rel="stylesheet">

    <link rel="stylesheet" href="/css/style.css">
</head>
<body>
    <main class="container">
        {{ block "main" . }}{{ end }}
    </main>

    <footer class="container">
        <p>&copy; {{ .Site.Params.author }}</p>
    </footer>
</body>
</html>
```

- [ ] **Step 2: Verify base template**

Run: `head -15 /Users/matthew-xh/Study/CS/my-academic-site/layouts/_default/baseof.html`
Expected: HTML starting with `<!DOCTYPE html>`

- [ ] **Step 3: Commit base template changes**

```bash
git add layouts/_default/baseof.html
git commit -m "refactor: remove navigation, add Lato font to base template

- Remove top navigation bar
- Add Lato font import
- Simplify structure for Jon Barron style

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 3: Homepage Layout

### Task 3: Rewrite Homepage with Jon Barron Style

**Files:**
- Rewrite: `layouts/index.html`

- [ ] **Step 1: Write the new homepage template**

```html
{{ define "main" }}
<table style="width:100%;max-width:800px;border:0px;border-spacing:0px;border-collapse:separate;margin-right:auto;margin-left:auto;">
<tbody>
    <!-- Header: Bio + Photo -->
    <tr style="padding:0px">
        <td style="padding:0px">
            <table style="width:100%;border:0px;border-spacing:0px;border-collapse:separate;margin-right:auto;margin-left:auto;">
            <tbody>
                <tr style="padding:0px">
                    <!-- Bio section (63%) -->
                    <td style="padding:2.5%;width:63%;vertical-align:middle">
                        <p class="name" style="text-align: center;">
                            {{ .Site.Title }}
                        </p>
                        <p>
                            {{ .Content }}
                        </p>
                        <p style="text-align:center">
                            <a href="mailto:{{ .Site.Params.email }}">Email</a> &nbsp;/&nbsp;
                            <a href="{{ .Site.Params.cv_url }}">CV</a> &nbsp;/&nbsp;
                            {{ with .Site.Params.scholar }}<a href="{{ . }}">Scholar</a> &nbsp;/&nbsp;{{ end }}
                            <a href="{{ .Site.Params.github }}">GitHub</a> &nbsp;/&nbsp;
                            <a href="/notes/">Notes</a> &nbsp;/&nbsp;
                            <a href="/paper-reading/">Paper Reading</a>
                        </p>
                    </td>
                    <!-- Photo section (37%) -->
                    <td style="padding:2.5%;width:37%;max-width:37%">
                        <div class="profile-photo">ZY</div>
                    </td>
                </tr>
            </tbody>
            </table>

            <!-- Research section -->
            <table style="width:100%;border:0px;border-spacing:0px;border-collapse:separate;margin-right:auto;margin-left:auto;">
            <tbody>
                <tr>
                    <td style="padding:16px;width:100%;vertical-align:middle">
                        <h2>Research</h2>
                        <p>
                            I'm interested in computer vision, 3D reconstruction, and generative AI.
                        </p>
                    </td>
                </tr>
            </tbody>
            </table>

            <!-- Publications -->
            <table style="width:100%;border:0px;border-spacing:0px 10px;border-collapse:separate;margin-right:auto;margin-left:auto;">
            <tbody>
                {{ $pubs := where .Site.RegularPages "Section" "publications" }}
                {{ range $pubs.ByDate.Reverse }}
                <tr onmouseout="{{ .File.BaseFileName }}_stop()" onmouseover="{{ .File.BaseFileName }}_start()" {{ if .Params.selected }}bgcolor="#ffffd0"{{ end }}>
                    <td style="padding:16px;width:20%;vertical-align:middle">
                        <div class="one">
                            {{ if .Params.hover_image }}
                            <div class="two" id='{{ .File.BaseFileName }}_image'>
                                <img src='{{ .Params.hover_image }}' width=100%>
                            </div>
                            {{ end }}
                            {{ if .Params.image }}
                            <img src='{{ .Params.image }}' width="160">
                            {{ else }}
                            <div style="width:160px;height:107px;background:#eee;"></div>
                            {{ end }}
                        </div>
                        {{ if .Params.hover_image }}
                        <script type="text/javascript">
                            function {{ .File.BaseFileName }}_start() {
                                document.getElementById('{{ .File.BaseFileName }}_image').style.opacity = "1";
                            }
                            function {{ .File.BaseFileName }}_stop() {
                                document.getElementById('{{ .File.BaseFileName }}_image').style.opacity = "0";
                            }
                            {{ .File.BaseFileName }}_stop()
                        </script>
                        {{ end }}
                    </td>
                    <td style="padding:20px;width:80%;vertical-align:middle">
                        <a href="{{ .Permalink }}">
                            <span class="papertitle">{{ .Title }}</span>
                        </a>
                        <br>
                        {{ .Params.authors | markdownify }}
                        <br>
                        <em>{{ .Params.venue }}</em>, {{ .Date.Format "2006" }}
                        <br>
                        {{ if .Params.paper }}<a href="{{ .Params.paper }}">paper</a>{{ end }}
                        {{ if and .Params.paper .Params.code }} / {{ end }}
                        {{ if .Params.code }}<a href="{{ .Params.code }}">code</a>{{ end }}
                        {{ if .Params.abstract }}
                        <p>{{ .Params.abstract | truncate 200 }}</p>
                        {{ end }}
                        {{ if or .Params.equal_contrib .Params.corresponding }}
                        <p style="font-size:13px;color:#666;font-style:italic;">
                            {{ if .Params.equal_contrib }}<sup>*</sup>Equal contribution{{ end }}
                            {{ if and .Params.equal_contrib .Params.corresponding }}&nbsp;&nbsp;{{ end }}
                            {{ if .Params.corresponding }}<sup>✉</sup>Corresponding author{{ end }}
                        </p>
                        {{ end }}
                    </td>
                </tr>
                {{ end }}
            </tbody>
            </table>
        </td>
    </tr>
</tbody>
</table>
{{ end }}
```

- [ ] **Step 2: Verify homepage template**

Run: `head -30 /Users/matthew-xh/Study/CS/my-academic-site/layouts/index.html`
Expected: Hugo template starting with `{{ define "main" }}`

- [ ] **Step 3: Commit homepage changes**

```bash
git add layouts/index.html
git commit -m "feat: Jon Barron style homepage

- Table-based layout with bio and photo placeholder
- All links in bio section (Email, CV, Scholar, GitHub, Notes, Paper Reading)
- Publications with thumbnails and hover effects
- Yellow highlight for selected papers
- Support for author contribution symbols

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 4: Notes Page and Content Updates

### Task 4: Update Notes Page with Intro Text

**Files:**
- Modify: `layouts/notes/list.html`

- [ ] **Step 1: Add intro text to notes template**

```html
{{ define "main" }}
<h1>Notes</h1>
<p class="notes-intro">
    Here I share my study notes on mathematics, computer vision, and related topics.
</p>
<div class="notes-list">
    {{ range .Pages.ByDate.Reverse }}
    <div class="note-item-container">
        <h3 class="note-item-title">
            <a href="{{ .Permalink }}">{{ .Title }}</a>
        </h3>
        <div class="note-item-meta">
            {{ .Date.Format "Jan 2, 2006" }}
            {{ if .Params.tags }}
            | {{ range $index, $tag := .Params.tags }}{{ if $index }}, {{ end }}{{ $tag }}{{ end }}
            {{ end }}
        </div>
        {{ if .Params.tldr }}
        <p class="note-item-summary">{{ .Params.tldr }}</p>
        {{ else }}
        <p class="note-item-summary">{{ .Summary | truncate 200 }}</p>
        {{ end }}
    </div>
    {{ end }}
</div>
{{ end }}
```

- [ ] **Step 2: Verify notes template**

Run: `head -10 /Users/matthew-xh/Study/CS/my-academic-site/layouts/notes/list.html`
Expected: Template starting with `{{ define "main" }}` and containing "notes-intro"

- [ ] **Step 3: Commit notes template changes**

```bash
git add layouts/notes/list.html
git commit -m "feat: add intro text to notes page

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

### Task 5: Update Publication Front Matter

**Files:**
- Modify: `content/publications/g4splat.md`

- [ ] **Step 1: Update author format with contribution symbols**

```yaml
---
title: "G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior"
date: 2025-10-14
authors: "Junfeng Ni<sup>*</sup>, Yixin Chen<sup>*</sup>, **Zhifei Yang**, Yu Liu, Ruijie Lu, Song-Chun Zhu, Siyuan Huang<sup>✉</sup>"
venue: "ICLR"
year: 2026
paper: "https://arxiv.org/abs/2510.12099"
code: "https://github.com/DaLi-Jack/G4Splat"
image: "/images/g4splat.png"
selected: true
equal_contrib: true
corresponding: true
abstract: "Despite recent advances in leveraging generative prior from pre-trained diffusion models for 3D scene reconstruction, existing methods still face two critical limitations. First, due to the lack of reliable geometric supervision, they struggle to produce high-quality reconstructions even in observed regions, let alone in unobserved areas. Second, they lack effective mechanisms to mitigate multiview inconsistencies in the generated images, leading to severe shape–appearance ambiguities and degraded scene geometry."
---
```

- [ ] **Step 2: Verify publication content**

Run: `head -15 /Users/matthew-xh/Study/CS/my-academic-site/content/publications/g4splat.md`
Expected: YAML front matter with updated authors field

- [ ] **Step 3: Commit publication changes**

```bash
git add content/publications/g4splat.md
git commit -m "feat: add author contribution symbols to G4Splat

- Add * for equal contribution
- Add ✉ for corresponding author
- Update venue format

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

### Task 6: Update Hugo Config

**Files:**
- Modify: `hugo.toml`

- [ ] **Step 1: Add scholar URL and update config**

```toml
baseURL = 'https://example.org/'
languageCode = 'en-us'
title = 'Zhifei Yang'

[params]
    author = "Zhifei Yang"
    description = "Undergraduate Student at Peking University"
    email = "zhifei.yeung@gmail.com"
    github = "https://github.com/isxiaohe"
    cv_url = "/cv.pdf"
    scholar = "https://scholar.google.com/citations?user=YOUR_ID"
```

- [ ] **Step 2: Verify config**

Run: `cat /Users/matthew-xh/Study/CS/my-academic-site/hugo.toml`
Expected: Config without menu section, with scholar parameter

- [ ] **Step 3: Commit config changes**

```bash
git add hugo.toml
git commit -m "refactor: remove menu, add scholar URL placeholder

- Remove navigation menu (links now in bio section)
- Add scholar URL parameter

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 5: Build and Test

### Task 7: Build and Verify

- [ ] **Step 1: Build the Hugo site**

Run: `cd /Users/matthew-xh/Study/CS/my-academic-site && hugo`
Expected: Site builds without errors

- [ ] **Step 2: Check generated homepage**

Run: `head -50 /Users/matthew-xh/Study/CS/my-academic-site/public/index.html`
Expected: HTML with Jon Barron style elements

- [ ] **Step 3: Verify CSS is copied**

Run: `head -10 /Users/matthew-xh/Study/CS/my-academic-site/public/css/style.css`
Expected: CSS content with Lato font import

- [ ] **Step 4: Final commit if any changes**

```bash
git status
# If public/ changes, they're generated - no need to commit
```

---

## Summary

After completing this plan:
1. Website will have Jon Barron style (Lato font, 800px width, blue→orange links)
2. Homepage will show bio with placeholder photo and all links
3. Publications will have thumbnails with hover effects
4. Notes page will have intro text
5. Author contribution symbols will display correctly
