#based on https://github.com/lemire/SIMDCompressionAndIntersection/blob/master/tools/clang-format.sh
STYLE=$(which clang-format-4.0)

RE=0
BASE=$(git rev-parse --show-toplevel)

ALLFILES=$(git ls-tree --full-tree --name-only  -r HEAD tiny_dnn test examples| grep -e ".*\.\(c\|h\|cc\|cpp\|hh\)\$")
for FILE in $ALLFILES; do
    $STYLE $BASE/$FILE | cmp -s $BASE/$FILE -
    if [ $? -ne 0 ]; then
        echo "$BASE/$FILE does not respect the coding style. Formatting. " >&2
        $STYLE -i $BASE/$FILE
        RE=1
    fi
done

exit $RE