#!/bin/bash

# Variables
INPUT_LOCAL="inpt.txt"
INPUT_HDFS_DIR="/input"
OUTPUT_HDFS_DIR1="/output1"
OUTPUT_HDFS_DIR2="/output2"
CONTAINER_NAME="hadoop"  # Replace with your container name
CONTAINER_WORKDIR="/opt/hadoop/code"

# Clean previous HDFS files
hdfs dfs -rm -r -f $INPUT_HDFS_DIR $OUTPUT_HDFS_DIR1 $OUTPUT_HDFS_DIR2
hdfs dfs -mkdir -p $INPUT_HDFS_DIR
hdfs dfs -put $CONTAINER_WORKDIR/$INPUT_LOCAL $INPUT_HDFS_DIR

# Run streaming job from inside code directory
hadoop jar /opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.6.jar \
  -input $INPUT_HDFS_DIR \
  -output $OUTPUT_HDFS_DIR1 \
  -mapper "python3 mapper_anagram.py" \
  -reducer "python3 reducer_anagram.py" \
  -file mapper_anagram.py \
  -file reducer_anagram.py

hadoop jar /opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.6.jar \
  -input $OUTPUT_HDFS_DIR1 \
  -output $OUTPUT_HDFS_DIR2 \
  -mapper "python3 mapper_anagram_2.py" \
  -reducer "python3 reducer_anagram_2.py" \
  -file mapper_anagram_2.py \
  -file reducer_anagram_2.py

# Step 3: Fetch result back
hdfs dfs -cat $OUTPUT_HDFS_DIR2/part-00000 > $CONTAINER_WORKDIR/output_anagram.txt
echo "MapReduce job complete. Output saved to output_anagram.txt"