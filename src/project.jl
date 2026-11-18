# Repository paths and command-line flags. Include from every script before any `using`.

const PROJECT_ROOT = dirname(@__DIR__)

# always use the package versions of this repository (Project.toml, Manifest.toml), from any folder
import Pkg
Pkg.activate(PROJECT_ROOT; io=devnull)

hasflag(name, args=ARGS) = ("--" * name) in args
function flagvalue(name, default, args=ARGS)
    prefix = "--" * name * "="
    i = findfirst(a -> startswith(a, prefix), args)
    return i === nothing ? default : args[i][length(prefix)+1:end]
end

projectdir(args...) = joinpath(PROJECT_ROOT, args...)
assetsdir(args...) = projectdir("assets", args...)
published_dir(args...) = projectdir("data", "published", args...)

"data/generated, or data/generated/TAG with --tag=TAG (--smoke runs use the tag \"smoke\")."
function generated_dir(args...)
    tag = flagvalue("tag", hasflag("smoke") ? "smoke" : "")
    return isempty(tag) ? projectdir("data", "generated", args...) : projectdir("data", "generated", tag, args...)
end

"Data to read: --data=published (default), generated or a folder; with --simulate, the data just generated."
function data_dir(args...)
    source = hasflag("simulate") ? "generated" : flagvalue("data", "published")
    base = source == "published" ? published_dir() : source == "generated" ? generated_dir() : abspath(source)
    return joinpath(base, args...)
end

"Self-couplings of the simulations: --s=published (default), generated or a file."
function self_couplings_file()
    s = flagvalue("s", "published")
    return s == "published" ? published_dir("self_couplings.txt") :
           s == "generated" ? generated_dir("self_couplings.txt") : abspath(s)
end

"""
Path for new data, data/generated[/TAG]/name. If it exists the script stops, unless --force (the old data are
then deleted). This is the only place where the scripts write data.
"""
function new_data_path(name)
    path = generated_dir(name)
    if ispath(path)
        hasflag("force") || error("$path already exists: add --force to replace it, or --tag=NAME to write to data/generated/NAME/")
        rm(path; recursive=true)
    end
    mkpath(dirname(path))
    return path
end
new_data_dir(name) = mkpath(new_data_path(name))

"Output prefix of a figure: --out=PREFIX, or figures/<data source>/name."
function figure_path(name)
    out = flagvalue("out", "")
    isempty(out) || return out
    d = data_dir()
    source = startswith(d, projectdir("data")) ? relpath(d, projectdir("data")) : basename(rstrip(d, '/'))
    return joinpath(mkpath(projectdir("figures", source)), name)
end
figures_dir(args...) = mkpath(projectdir("figures", args...))
