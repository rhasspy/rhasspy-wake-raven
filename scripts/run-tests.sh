#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

venv="${src_dir}/.venv"
if [[ -d "${venv}" ]]; then
    source "${venv}/bin/activate"
fi

# -----------------------------------------------------------------------------

coverage run "--source=$1" -m pytest
coverage report -m
coverage xml

# -----------------------------------------------------------------------------

# Test WAV files
temp_dir="$(mktemp -d)"
function cleanup {
    rm -rf "${temp_dir}"
}

trap cleanup EXIT

# Set up Mycroft-style test directories
# test/wake-word, test/not-wake-word
export PATH="${src_dir}/bin:${PATH}"

function test_f1 {
    # keyword names in etc/
    # e.g., okay-rhasspy hey-mycroft
    positive_name="$1"
    negative_name="$2"

    test_dir="${temp_dir}/${positive_name}"
    positive_dir="${test_dir}/wake-word"
    negative_dir="${test_dir}/not-wake-word"
    results="${test_dir}/results.json"

    # Copy WAV files into directory structure
    mkdir -p "${positive_dir}" "${negative_dir}"
    cp "${src_dir}/etc/${positive_name}/"*.wav "${positive_dir}/"
    cp "${src_dir}/etc/${negative_name}/"*.wav "${negative_dir}/"

    python3 "${src_dir}/bin/test-raven.py" \
            --keyword "${src_dir}/etc/${positive_name}" \
            --average-templates \
            --test-directory "${test_dir}" \
            > "${results}"

    # Format F1 score
    f1="$(jq -r .summary.f1_score < "${results}")"
    f1="$(printf '%0.1f' "${f1}")"

    # Should be perfect
    if [[ ! "${f1}" == "1.0" ]]; then
        echo "Got F1 of ${f1}"
        cat "${results}"
        exit 1
    fi

    echo "${positive_name}/${negative_name} OK"
}

test_f1 'okay-rhasspy' 'hey-mycroft'
test_f1 'hey-mycroft' 'okay-rhasspy'

# -----------------------------------------------------------------------------

echo "OK"
