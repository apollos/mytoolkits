#!/bin/bash

OUTPUT="/tmp/nvidia-smi.out.$$"

function getinfo()
{
    nvidia-smi -q -d UTILIZATION >"$OUTPUT"
    if (( $? != 0 ))
        then
        echo "nvidia-smi: failed $?" && exit 1
    fi
    # Get the number of how many GPUs do you have
    declare -i COUNT="$(awk '/Attached GPUs/ { print $NF }' "$OUTPUT")"
    # Get the value of the `Memory utilization', and put them in an array
    declare -a MEMORYUti=( $(awk '/Memory +:/ { print $(NF - 1) }' "$OUTPUT") )
    # Get the value of the `GPU utilization', and put them in an array
    declare -a GPUUti=( $(awk '/Gpu +:/ { print $(NF - 1) }' "$OUTPUT") )

    declare -i i
    for (( i = 0; i < COUNT ; ++i ))
    do
	    echo "GPU ${i+1}: Memmory Utilization ${MEMORYUti[$i]}%"
        echo "       GPU     Utilization ${GPUUti[$i]}%"
    done
}


# Remove the output file just before the shell script exit
trap "rm -f \"$OUTPUT\"" 0
# Check if the command nvidia-smi exist, otherwise die.
type nvidia-smi >/dev/null 2>&1 ||
    (echo ""${0##*/}": nvidia-smi: command not found" >&2 && exit 1)

while (( 1 ))
do
    getinfo
    sleep 3
done
