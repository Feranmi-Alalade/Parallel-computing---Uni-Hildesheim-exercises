#!/bin/bash

# Variables
INPUT_LOCAL="inpt.txt"
INPUT_HDFS_DIR="/input"
OUTPUT_HDFS_DIR="/output"
CONTAINER_NAME="hadoop"  # Replace with your container name
CONTAINER_WORKDIR="/opt/hadoop/code"
OUTPUT_LOCAL="hadoop_code/output.txt"

# Clean previous HDFS files
docker exec -it $CONTAINER_NAME hdfs dfs -rm -r -f $INPUT_HDFS_DIR $OUTPUT_HDFS_DIR
docker exec -it $CONTAINER_NAME hdfs dfs -mkdir -p $INPUT_HDFS_DIR
docker exec -it $CONTAINER_NAME hdfs dfs -put $CONTAINER_WORKDIR/$INPUT_LOCAL $INPUT_HDFS_DIR

# Run streaming job from inside code directory
docker exec -w $CONTAINER_WORKDIR -it $CONTAINER_NAME hadoop jar /opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.6.jar \
  -input $INPUT_HDFS_DIR \
  -output $OUTPUT_HDFS_DIR \
  -mapper "python3 mapper.py" \
  -reducer "python3 reducer.py" \
  -file mapper.py \
  -file reducer.py

# Step 3: Fetch result back
docker exec -it $CONTAINER_NAME hdfs dfs -cat $OUTPUT_HDFS_DIR/part-00000 > $OUTPUT_LOCAL

echo "MapReduce job complete. Output saved to $OUTPUT_LOCAL"