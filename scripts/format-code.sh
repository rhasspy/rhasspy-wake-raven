#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

venv="${src_dir}/.venv"
if [[ -d "${venv}" ]]; then
    echo "Using virtual environment at ${venv}"
    source "${venv}/bin/activate"
fi

dir_name="$(basename "${src_dir}")"
python_name="$(echo "${dir_name}" | sed -e 's/-//' | sed -e 's/-/_/g')"
python_files=(
    "${src_dir}/${python_name}"/*.py
    "${src_dir}/setup.py"
)

# -----------------------------------------------------------------------------

black "${python_files[@]}"
isort "${python_files[@]}"

# -----------------------------------------------------------------------------

echo "OK"
