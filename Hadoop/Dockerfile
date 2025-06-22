FROM openjdk:11

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip sudo ssh && \
    apt-get clean

# Set Hadoop version and environment
ENV HADOOP_VERSION=3.3.6
ENV HADOOP_HOME=/opt/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

# Download and extract Hadoop
RUN curl -O https://downloads.apache.org/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz && \
    tar -xzvf hadoop-${HADOOP_VERSION}.tar.gz && \
    mv hadoop-${HADOOP_VERSION} $HADOOP_HOME && \
    rm hadoop-${HADOOP_VERSION}.tar.gz

# Set up passwordless SSH
RUN ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    chmod 600 ~/.ssh/authorized_keys && \
    echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

# Start sshd and format HDFS
RUN mkdir -p /var/run/sshd && \
    mkdir -p /tmp/hadoop-root/dfs/name

# Copy Hadoop configuration
COPY core-site.xml $HADOOP_HOME/etc/hadoop/
COPY hdfs-site.xml $HADOOP_HOME/etc/hadoop/
COPY env_vars.sh /etc/profile.d/hadoop_env.sh
RUN chmod +x /etc/profile.d/hadoop_env.sh

RUN echo "source /etc/profile.d/hadoop_env.sh" >> /root/.bashrc


# Format HDFS
RUN $HADOOP_HOME/bin/hdfs namenode -format

# Expose ports
EXPOSE 9870 9000 22

CMD bash -c "source /etc/profile.d/hadoop_env.sh && \
             service ssh start && \
             $HADOOP_HOME/sbin/start-dfs.sh && \
             tail -f /dev/null"
