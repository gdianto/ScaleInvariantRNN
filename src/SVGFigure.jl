# Minimal SVG figure builder: coordinates in mm from the top-left corner, line widths and font sizes in pt, Arial
# text with `_{...}`/`^{...}` sub/superscripts (never below MIN_PT); PDF/PNG exported with Inkscape.
module SVGFigure

using Base64, Printf

export Figure, Axes, xmm, ymm, line!, polygon!, rect!, circle!, text!, image!, raw!, contour!, embed!, place!, png_bytes,
       plot!, band!, steps!, scatter!, xaxis!, yaxis!, save, PT

const PT = 25.4 / 72                                  # mm per pt
fmt(x::Real) = @sprintf("%.4f", x)
colorstr(c::AbstractString) = c

mutable struct Figure
    w::Float64; h::Float64
    defs::IOBuffer; body::IOBuffer
    nclip::Int
end
Figure(w, h) = Figure(w, h, IOBuffer(), IOBuffer(), 0)

# Axes: box [left, right] x [top, bottom] in mm, data limits, linear or log10 scales
struct Axes
    fig::Figure
    left::Float64; right::Float64; top::Float64; bottom::Float64
    xlim::Tuple{Float64,Float64}; ylim::Tuple{Float64,Float64}
    xlog::Bool; ylog::Bool
    clip::String
end
function Axes(fig::Figure; left, right, top, bottom, xlim, ylim, xscale=:identity, yscale=:identity)
    fig.nclip += 1
    id = "clip$(fig.nclip)"
    print(fig.defs, "<clipPath id=\"$id\"><rect x=\"$(fmt(left))\" y=\"$(fmt(top))\" width=\"$(fmt(right-left))\" height=\"$(fmt(bottom-top))\"/></clipPath>\n")
    Axes(fig, left, right, top, bottom, Float64.(xlim), Float64.(ylim), xscale == :log10, yscale == :log10, id)
end
# Axes from measured mappings mm = c0 + c1*u (u = value or log10 value); the box is used for clipping
function Axes(fig::Figure, box::NTuple{4,Real}, xmap::NTuple{2,Real}, ymap::NTuple{2,Real}; xscale=:identity, yscale=:identity)
    left, right, top, bottom = box
    u(m, map) = (m - map[1]) / map[2]
    v(a, lg) = lg ? 10.0^a : a
    xl, yl = xscale == :log10, yscale == :log10
    return Axes(fig; left, right, top, bottom, xscale, yscale,
                xlim=(v(u(left, xmap), xl), v(u(right, xmap), xl)), ylim=(v(u(bottom, ymap), yl), v(u(top, ymap), yl)))
end
_t(v, lg) = lg ? log10(v) : v
function xmm(ax::Axes, x)
    a, b = _t(ax.xlim[1], ax.xlog), _t(ax.xlim[2], ax.xlog)
    return ax.left + (_t(x, ax.xlog) - a) / (b - a) * (ax.right - ax.left)
end
function ymm(ax::Axes, y)
    a, b = _t(ax.ylim[1], ax.ylog), _t(ax.ylim[2], ax.ylog)
    return ax.bottom - (_t(y, ax.ylog) - a) / (b - a) * (ax.bottom - ax.top)
end

function _stroke(color, width, opacity, dash, cap, join)
    s = "fill=\"none\" stroke=\"$(colorstr(color))\" stroke-width=\"$(fmt(width*PT))\" stroke-linecap=\"$cap\" stroke-linejoin=\"$join\""
    opacity < 1 && (s *= " stroke-opacity=\"$(fmt(opacity))\"")
    dash === nothing || (s *= " stroke-dasharray=\"" * join_dash(dash) * "\"")
    return s
end
join_dash(d) = Base.join((fmt(v*PT) for v in d), ",")
_clip(clip) = clip === nothing ? "" : " clip-path=\"url(#$clip)\""
pts(xs, ys) = Base.join((fmt(x) * "," * fmt(y) for (x, y) in zip(xs, ys) if isfinite(x) && isfinite(y)), " ")

function line!(fig::Figure, xs, ys; color="#000000", width=0.64, opacity=1.0, dash=nothing, cap="round", join="round", clip=nothing)
    print(fig.body, "<polyline points=\"$(pts(xs, ys))\" $(_stroke(color, width, opacity, dash, cap, join))$(_clip(clip))/>\n")
end
function polygon!(fig::Figure, xs, ys; fill="#000000", fill_opacity=1.0, stroke="none", width=0.0, clip=nothing)
    st = stroke == "none" ? "" : " stroke=\"$stroke\" stroke-width=\"$(fmt(width*PT))\""
    print(fig.body, "<polygon points=\"$(pts(xs, ys))\" fill=\"$fill\" fill-opacity=\"$(fmt(fill_opacity))\"$st$(_clip(clip))/>\n")
end
function rect!(fig::Figure, x, y, w, h; fill="none", fill_opacity=1.0, stroke="none", width=0.64, stroke_opacity=1.0, rx=0.0, ry=rx, clip=nothing)
    st = stroke == "none" ? "" : " stroke=\"$stroke\" stroke-width=\"$(fmt(width*PT))\" stroke-opacity=\"$(fmt(stroke_opacity))\""
    r = rx > 0 ? " rx=\"$(fmt(rx))\" ry=\"$(fmt(ry))\"" : ""
    print(fig.body, "<rect x=\"$(fmt(x))\" y=\"$(fmt(y))\" width=\"$(fmt(w))\" height=\"$(fmt(h))\"$r fill=\"$fill\" fill-opacity=\"$(fmt(fill_opacity))\"$st$(_clip(clip))/>\n")
end
function circle!(fig::Figure, x, y, r; fill="#000000", fill_opacity=1.0, stroke="none", width=0.0, stroke_opacity=1.0, clip=nothing)
    st = stroke == "none" ? "" : " stroke=\"$stroke\" stroke-width=\"$(fmt(width*PT))\" stroke-opacity=\"$(fmt(stroke_opacity))\""
    print(fig.body, "<circle cx=\"$(fmt(x))\" cy=\"$(fmt(y))\" r=\"$(fmt(r))\" fill=\"$fill\" fill-opacity=\"$(fmt(fill_opacity))\"$st$(_clip(clip))/>\n")
end

xmlesc(s) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")
const MIN_PT = 5.0                                    # smallest text of the figures (pt), sub/superscripts included
# "g_{b}=0.1" -> text with sub/superscript tspans (65% of `size` pt, at least MIN_PT)
function _rich(s::AbstractString, size=7.0)
    pct = max(65, ceil(Int, 100 * MIN_PT / size))
    out = IOBuffer(); i = firstindex(s)
    while i <= lastindex(s)
        c = s[i]
        if (c == '_' || c == '^') && i < lastindex(s) && s[nextind(s, i)] == '{'
            j = findnext('}', s, i)
            inner = s[nextind(s, nextind(s, i)):prevind(s, j)]
            print(out, "<tspan style=\"font-size:$pct%;baseline-shift:$(c == '_' ? "sub" : "super")\">", xmlesc(inner), "</tspan>")
            i = nextind(s, j)
        else
            print(out, xmlesc(string(c))); i = nextind(s, i)
        end
    end
    return String(take!(out))
end
# anchor: "start" | "middle" | "end"; (x, y) is the baseline point; rotate in degrees (counter-clockwise = -90)
function text!(fig::Figure, x, y, s; size=7.0, anchor="start", weight="normal", rotate=0, color="#000000", family="Arial", style="normal")
    tf = rotate == 0 ? "" : " transform=\"rotate($(fmt(rotate)) $(fmt(x)) $(fmt(y)))\""
    w = weight == "normal" ? "" : "font-weight:$weight;"
    it = style == "normal" ? "" : "font-style:$style;"
    print(fig.body, "<text x=\"$(fmt(x))\" y=\"$(fmt(y))\" style=\"font-family:$family;font-size:$(fmt(size*PT))px;$w$it" *
          "text-anchor:$anchor;fill:$color\"$tf>", _rich(s, size), "</text>\n")
end
# PNG (bytes or file path) stretched to the box; pixelated=true keeps hard pixel edges (heatmaps)
function image!(fig::Figure, x, y, w, h, png; pixelated=false, clip=nothing)
    data = png isa AbstractString ? read(png) : png
    r = pixelated ? " style=\"image-rendering:optimizeSpeed\"" : ""
    print(fig.body, "<image x=\"$(fmt(x))\" y=\"$(fmt(y))\" width=\"$(fmt(w))\" height=\"$(fmt(h))\" preserveAspectRatio=\"none\"$r$(_clip(clip)) ",
          "xlink:href=\"data:image/png;base64,", base64encode(data), "\"/>\n")
end
raw!(fig::Figure, s::AbstractString) = print(fig.body, s, "\n")

# Minimal PNG encoder (8-bit RGB, uncompressed) for heatmaps; rgb[i, j] = (r, g, b) of row i (top to bottom), column j
function png_bytes(rgb::AbstractMatrix{NTuple{3,UInt8}})
    h, w = size(rgb)
    crc_table = [begin c = UInt32(n); for _ in 1:8; c = isodd(c) ? 0xedb88320 ⊻ (c >> 1) : c >> 1; end; c end for n in 0:255]
    crc(bytes) = ~foldl((c, b) -> crc_table[((c ⊻ b) & 0xff) + 1] ⊻ (c >> 8), bytes; init=0xffffffff)
    be32(x) = UInt8[(x >> 24) & 0xff, (x >> 16) & 0xff, (x >> 8) & 0xff, x & 0xff]
    chunk(t, d) = vcat(be32(UInt32(length(d))), Vector{UInt8}(t), d, be32(crc(vcat(Vector{UInt8}(t), d))))
    raw = UInt8[]
    for i in 1:h
        push!(raw, 0x00)
        for j in 1:w; append!(raw, rgb[i, j]); end
    end
    z = UInt8[0x78, 0x01]
    for (k, i0) in enumerate(1:65535:length(raw))
        blk = raw[i0:min(i0 + 65534, length(raw))]; n = length(blk)
        push!(z, i0 + 65535 > length(raw) ? 0x01 : 0x00)
        append!(z, UInt8[n & 0xff, n >> 8, ~UInt16(n) & 0xff, (~UInt16(n)) >> 8], blk)
    end
    a, b = UInt32(1), UInt32(0)
    for x in raw; a = (a + x) % 65521; b = (b + a) % 65521; end
    append!(z, be32((b << 16) | a))
    ihdr = vcat(be32(UInt32(w)), be32(UInt32(h)), UInt8[8, 2, 0, 0, 0])
    return vcat(UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], chunk("IHDR", ihdr), chunk("IDAT", z), chunk("IEND", UInt8[]))
end

function plot!(ax::Axes, x, y; clip=true, kw...)
    line!(ax.fig, [xmm(ax, v) for v in x], [ymm(ax, v) for v in y]; clip=clip ? ax.clip : nothing, kw...)
end
function band!(ax::Axes, x, lo, hi; fill, fill_opacity=0.5, clip=true)
    xs = vcat([xmm(ax, v) for v in x], [xmm(ax, v) for v in reverse(x)])
    ys = vcat([ymm(ax, v) for v in lo], [ymm(ax, v) for v in reverse(hi)])
    polygon!(ax.fig, xs, ys; fill=fill, fill_opacity=fill_opacity, clip=clip ? ax.clip : nothing)
end
# step histogram: bin edges e (n+1) and heights h (n); filled down to `base` and/or outlined
function steps!(ax::Axes, e, h; color, width=0.63, fill=nothing, fill_opacity=0.35, base=0.0, clip=true)
    xs = Float64[]; ys = Float64[]
    for i in eachindex(h)
        push!(xs, e[i], e[i+1]); push!(ys, h[i], h[i])
    end
    if fill !== nothing
        band!(ax, xs, Base.fill(base, length(xs)), ys; fill=fill, fill_opacity=fill_opacity, clip=clip)
    end
    plot!(ax, vcat(e[1], xs, e[end]), vcat(base, ys, base); color=color, width=width, clip=clip, join="miter", cap="butt")
end
function scatter!(ax::Axes, x, y; r=0.3, fill="#000000", fill_opacity=1.0, stroke="none", width=0.0, clip=true)
    for (a, b) in zip(x, y)
        circle!(ax.fig, xmm(ax, a), ymm(ax, b), r; fill=fill, fill_opacity=fill_opacity, stroke=stroke, width=width,
                clip=clip ? ax.clip : nothing)
    end
end

# Iso-line of Z (Z[i, j] at (x[j], y[i])) at `level`, by marching squares
function contour!(ax::Axes, x, y, Z, level; color="#000000", width=0.64, opacity=1.0, clip=true)
    seg = IOBuffer()
    interp(a, b, za, zb) = a + (level - za) / (zb - za) * (b - a)
    for i in 1:length(y)-1, j in 1:length(x)-1
        z = (Z[i, j], Z[i, j+1], Z[i+1, j+1], Z[i+1, j])
        p = ((x[j], y[i]), (x[j+1], y[i]), (x[j+1], y[i+1]), (x[j], y[i+1]))
        pts_ = Tuple{Float64,Float64}[]
        for (a, b) in ((1, 2), (2, 3), (3, 4), (4, 1))
            if (z[a] < level) != (z[b] < level)
                push!(pts_, (interp(p[a][1], p[b][1], z[a], z[b]), interp(p[a][2], p[b][2], z[a], z[b])))
            end
        end
        for k in 1:2:length(pts_)-1
            print(seg, "M", fmt(xmm(ax, pts_[k][1])), ",", fmt(ymm(ax, pts_[k][2])), "L", fmt(xmm(ax, pts_[k+1][1])), ",", fmt(ymm(ax, pts_[k+1][2])))
        end
    end
    d = String(take!(seg))
    isempty(d) || print(ax.fig.body, "<path d=\"$d\" $(_stroke(color, width, opacity, nothing, "round", "round"))$(_clip(clip ? ax.clip : nothing))/>\n")
end

# Embed an SVG fragment (e.g. artwork extracted from a published figure) whose user coordinates map to the box
function embed!(fig::Figure, svgfile::AbstractString; x=0.0, y=0.0, w=fig.w, h=fig.h)
    s = read(svgfile, String)
    vb = match(r"viewBox=\"([^\"]+)\"", s).captures[1]
    a = findfirst('>', s[findfirst("<svg", s)[1]:end]) + findfirst("<svg", s)[1] - 1
    inner = s[a+1:findlast("</svg>", s)[1]-1]
    print(fig.body, "<svg x=\"$(fmt(x))\" y=\"$(fmt(y))\" width=\"$(fmt(w))\" height=\"$(fmt(h))\" viewBox=\"$vb\" ",
          "xmlns:inkscape=\"http://www.inkscape.org/namespaces/inkscape\" xmlns:sodipodi=\"http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd\">",
          inner, "</svg>\n")
end

# Place an SVG drawn in page coordinates (mm), e.g. a piece of a published figure, shifted by (dx, dy) mm
function place!(fig::Figure, svgfile::AbstractString, dx, dy)
    s = read(svgfile, String)
    a = findfirst('>', s[findfirst("<svg", s)[1]:end]) + findfirst("<svg", s)[1] - 1
    inner = s[a+1:findlast("</svg>", s)[1]-1]
    print(fig.body, "<g transform=\"translate($(fmt(dx)),$(fmt(dy)))\" ",
          "xmlns:inkscape=\"http://www.inkscape.org/namespaces/inkscape\" xmlns:sodipodi=\"http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd\">",
          inner, "</g>\n")
end

# Spine along the bottom (or top) with outward ticks and tick labels (baseline `labeloffset` mm below the spine)
function xaxis!(ax::Axes; ticks=Float64[], labels=string.(ticks), ticklen=0.526, width=0.63, size=7.0,
                labeloffset=2.78, spine=true, at=:bottom, labelanchor="middle", labeldx=0.0)
    y = at == :bottom ? ax.bottom : ax.top
    s = at == :bottom ? 1 : -1
    spine && line!(ax.fig, [ax.left, ax.right], [y, y]; width=width, cap="butt")
    for (t, l) in zip(ticks, labels)
        x = xmm(ax, t)
        line!(ax.fig, [x, x], [y, y + s*ticklen]; width=width, cap="butt")
        isempty(l) || text!(ax.fig, x + labeldx, y + s*labeloffset, l; size=size, anchor=labelanchor)
    end
end
# Spine along the left (or right) with outward ticks and right-aligned tick labels
function yaxis!(ax::Axes; ticks=Float64[], labels=string.(ticks), ticklen=0.526, width=0.63, size=7.0,
                labeloffset=1.1, baselineshift=0.85, spine=true, at=:left, labelanchor="end")
    x = at == :left ? ax.left : ax.right
    s = at == :left ? -1 : 1
    spine && line!(ax.fig, [x, x], [ax.bottom, ax.top]; width=width, cap="butt")
    for (t, l) in zip(ticks, labels)
        y = ymm(ax, t)
        line!(ax.fig, [x, x + s*ticklen], [y, y]; width=width, cap="butt")
        isempty(l) || text!(ax.fig, x + s*labeloffset, y + baselineshift, l; size=size, anchor=labelanchor)
    end
end

# Writes PREFIX.svg; with `formats` containing "pdf"/"png" also exports them with Inkscape (if installed)
function save(fig::Figure, prefix::AbstractString; formats=("svg", "pdf", "png"), dpi=300)
    svg = "$prefix.svg"
    open(svg, "w") do io
        print(io, """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="$(fmt(fig.w))mm" height="$(fmt(fig.h))mm" viewBox="0 0 $(fmt(fig.w)) $(fmt(fig.h))">
<defs>
""", String(take!(copy(fig.defs))), "</defs>\n<rect x=\"0\" y=\"0\" width=\"$(fmt(fig.w))\" height=\"$(fmt(fig.h))\" fill=\"#ffffff\"/>\n",
              String(take!(copy(fig.body))), "</svg>\n")
    end
    done = [svg]
    others = filter(!=("svg"), collect(formats))
    if !isempty(others)
        if Sys.which("inkscape") === nothing
            @warn "Inkscape not found: only $svg was written"
        else
            for f in others
                cmd = f == "png" ? `inkscape $svg --export-type=png --export-dpi=$dpi --export-background=white --export-filename=$prefix.png` :
                                   `inkscape $svg --export-type=$f --export-filename=$prefix.$f`
                run(pipeline(cmd; stdout=devnull, stderr=devnull)); push!(done, "$prefix.$f")
            end
        end
    end
    println("saved ", Base.join(done, ", "))
    return done
end

end
