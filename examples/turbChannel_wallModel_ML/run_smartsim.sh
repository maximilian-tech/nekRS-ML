#!/bin/bash
copydir_unique() {
    local src="$1"
    local dst="$2"

    if [[ -z "$src" || -z "$dst" ]]; then
        printf 'usage: copydir_unique <source_dir> <target_dir>\n' >&2
        return 2
    fi

    if [[ ! -d "$src" ]]; then
        printf 'error: source directory does not exist: %s\n' "$src" >&2
        return 1
    fi

    local final_dst="$dst"
    local n=1

    while [[ -e "$final_dst" ]]; do
        final_dst="${dst}_$n"
        ((n++))
    done

    cp -a -- "$src" "$final_dst"
}

export CWD=$PWD


sed -i 's|^target_compile_definitions(udf PUBLIC KASANE=1)|#target_compile_definitions(udf PUBLIC KASANE=1)|g'  udf.cmake
sed -i 's|^#target_compile_definitions(udf PUBLIC SMARTREDIS=1)|target_compile_definitions(udf PUBLIC SMARTREDIS=1)|g' udf.cmake

export REPLAY_ENABLE=no


export DEFAULT_NUM_RANKS=6
export PALS_LOCAL_SIZE=${SIM_RANKS:-${DEFAULT_NUM_RANKS}}

export REPS=3

for idx in $(seq "$REPS"); do
  rm -rf .cache/udf
  killall redis-server 2>/dev/null || true

  python3 ./ssim_driver.py

  cd $CWD
  copydir_unique nekRS-ML nekRS-ML_smartredis
done


