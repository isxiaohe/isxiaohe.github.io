# Website Redesign: Jon Barron Style

## Overview

Transfer the current Hugo-based academic website to match Jon Barron's academic website style (jonbarron.github.io).

## Design Decisions

### Style Adoption
- **Full adoption** of Jon Barron style with hybrid navigation structure

### Typography
- **Font:** Lato (Google Fonts)
- **Body size:** 14px
- **Name title:** 32px, normal weight
- **Paper titles:** 14px, bold weight

### Layout
- **Max width:** 800px (centered)
- **Table-based layout** for header section (bio + photo)
- **Flex layout** for publication cards

### Colors
- **Primary link:** #1772d0 (blue)
- **Link hover:** #f09228 (orange)
- **Highlighted papers:** #ffffd0 (yellow background)
- **Body text:** #222222
- **Muted text:** #666666

### Profile Photo
- **Placeholder:** Circular div with gradient background
- **Size:** 160px × 160px
- **Content:** Initials "ZY" in white, 48px bold

### Publications
- **Thumbnail size:** 160px width
- **Hover effects:** Before/after image swap or video play on hover
- **Highlighted papers:** Yellow background (#ffffd0)
- **Layout:** Thumbnail on left, paper info on right

### Navigation
- **No top navigation bar** - pure Jon Barron style single-page feel
- **All links in bio section:** Email / CV / Scholar / GitHub / Notes / Paper Reading
- Centered horizontal link bar below bio text

### Author Symbols
- **Equal contribution:** Asterisk (*)
- **Corresponding author:** Envelope (✉)
- **Legend:** Display below author list in italic

## Site Structure

### Homepage (`/`)
1. Header section with name, bio, all links (including Notes & Paper Reading), and placeholder photo
2. Research interests section
3. All publications with thumbnails and hover effects

### Sub-pages
- `/notes/` - Notes page with introductory text: "Here I share my study notes on mathematics, computer vision, and related topics."
- `/paper-reading/` - Paper reading page

## Files to Modify

### Layouts
- `layouts/index.html` - Complete rewrite for Jon Barron style homepage
- `layouts/_default/baseof.html` - Update base template with Lato font, remove navigation
- `layouts/notes/list.html` - Add introductory text for notes section

### Styles
- `static/css/style.css` - Complete rewrite:
  - Lato font import
  - Table-based header layout
  - Publication card styles with hover effects
  - Link color scheme
  - Responsive adjustments

### Content
- `content/_index.md` - Update bio text if needed
- `content/publications/*.md` - Add image paths for thumbnails

### Assets
- `static/images/` - Add publication thumbnail images

## Publication Front Matter

Add these optional fields to publication markdown files:

```yaml
---
title: "Paper Title"
date: 2026-01-01
authors: "Author 1*, Author 2*, Author 3"
venue: "Conference"
paper: "https://arxiv.org/..."
code: "https://github.com/..."
image: "/images/paper-thumb.jpg"  # Thumbnail image
hover_image: "/images/paper-hover.jpg"  # Optional: image shown on hover
hover_video: "/images/paper-demo.mp4"  # Optional: video shown on hover
selected: true  # Highlight with yellow background
equal_contrib: [0, 1]  # Indices of equal contribution authors
corresponding: [3]  # Indices of corresponding authors
---
```

## Implementation Notes

1. Keep Hugo as the static site generator (don't switch to plain HTML)
2. Use Hugo templates to generate the Jon Barron-style HTML
3. Publication thumbnails will use CSS hover effects (opacity transition)
4. No navigation bar - all links (Email, CV, Scholar, GitHub, Notes, Paper Reading) in bio section
5. Maintain existing content structure for Notes and Paper Reading
6. Add intro text on Notes page to introduce viewers to the content

## Example Publication Card HTML

```html
<tr onmouseout="paper_stop()" onmouseover="paper_start()" bgcolor="#ffffd0">
  <td style="padding:16px;width:20%;vertical-align:middle">
    <div class="one">
      <div class="two" id='paper_image'>
        <img src='hover-image.jpg' width=100%>
      </div>
      <img src='thumb.jpg' width="160">
    </div>
    <script>
      function paper_start() {
        document.getElementById('paper_image').style.opacity = "1";
      }
      function paper_stop() {
        document.getElementById('paper_image').style.opacity = "0";
      }
      paper_stop()
    </script>
  </td>
  <td style="padding:20px;width:80%;vertical-align:middle">
    <a href="project-url">
      <span class="papertitle">Paper Title</span>
    </a>
    <br>
    Author 1*, Author 2*, <strong>Your Name</strong>, Author 3✉
    <br>
    <em>Conference</em>, 2026
    <br>
    <a href="paper-url">paper</a> / <a href="code-url">code</a>
    <p>Description text...</p>
  </td>
</tr>
```
