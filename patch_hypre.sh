

find 3rd_party/hypre/src/IJ_mv/ -type f -exec sed -E -i \
  -e '1 s/^make_reverse_iterator([[:space:]]*)\(/thrust::make_reverse_iterator\1(/' \
  -e 's/([^[:alnum:]_:])make_reverse_iterator([[:space:]]*)\(/\1thrust::make_reverse_iterator\2(/g' \
  {} +
