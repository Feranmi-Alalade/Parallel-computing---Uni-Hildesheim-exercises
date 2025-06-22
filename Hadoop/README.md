# Hadoop with Docker

## Preliminaries
### Docker
You need docker installed on your laptop. \
The installation process is different for each system:
- Linux: `sudo apt install docker`
- Windows: [Docker Windows Installation](https://docs.docker.com/desktop/install/windows-install/)
- Mac: [Docker MacOS Installation](https://docs.docker.com/desktop/setup/install/mac-install/)

You can check whether Docker is correctly installed by opening a terminal and trying to execute 
```
docker version
```

It should display a line like this:
```
(Hadoop-Project) abdelmalak@abdelmalak:~/Teaching/SoSe25/PycharmProjects/Hadoop_example$ docker version
Client: Docker Engine - Community
 Version:           28.1.1
 API version:       1.49
 Go version:        go1.23.8
 Git commit:        4eba377
 Built:             Fri Apr 18 09:52:10 2025
 OS/Arch:           linux/amd64
 Context:           default
```

### Docker-Compose
Docker-compose is an extension to Docker that allows for easy starting and controlling of Docker containers. \
You need to install is separately on your laptop. \
The installation process is different for each system:
- Linux: `sudo apt install docker-compose`
- Windows: [Docker Compose Windows Installation](https://docs.docker.com/desktop/setup/install/windows-install/)
- Mac: [Docker Compose MacOS Installation](https://docs.docker.com/desktop/setup/install/mac-install/)

You can check whether docker-compose is correctly installed by opening a terminal and trying to execute 
```
docker-compose version  
```

It should display a line like this:
```
(Hadoop-Project) abdelmalak@abdelmalak:~/Teaching/SoSe25/PycharmProjects/Hadoop_example$ docker-compose --version
docker-compose version 1.29.2, build unknown
```

## Starting the Docker Image
1. Open a terminal on your laptop (You can use the terminal inside PyCharm or VSCode)
2. Navigate to the `hadoop_docker_example` folder
3. Execute `docker-compose up --build`
   - The first time you do this will probably take a few minutes
   - If successful, the terminal will display 
   ```
    Starting hadoop ... done
    Attaching to hadoop
    hadoop    | Starting OpenBSD Secure Shell server: sshd.
    hadoop    | Starting namenodes on [localhost]
    hadoop    | Starting datanodes
    hadoop    | Starting secondary namenodes [eeace327577b]
   ```
**Note** : The current setup, I made that it starts all the namenode and datanode directly from the DockerFile which is why after attaching hadoop you see the lines realted to starting namenode, secondary namenode and datanode.
4. Make sure that the container stays running
   - You don't see a line like `hadoop exited with code 0`
   - You see the container when executing `docker ps`
   ```
    (Hadoop-Project) abdelmalak@abdelmalak:~/Teaching/SoSe25/PycharmProjects/Hadoop_example$ docker ps
    CONTAINER ID   IMAGE                   COMMAND                  CREATED        STATUS          PORTS                                                                                                                               NAMES
    eeace327577b   hadoop_example_hadoop   "/bin/sh -c 'bash -c…"   24 hours ago   Up 28 seconds   0.0.0.0:9000->9000/tcp, [::]:9000->9000/tcp, 0.0.0.0:9870->9870/tcp, [::]:9870->9870/tcp, 0.0.0.0:2222->22/tcp, [::]:2222->22/tcp   hadoop
   ```

## Connecting to the Docker Image
1. Open a **second** terminal (PyCharm and VSCode allow you to open multiple terminals as well)
2. Find the container_id of your docker image via `docker ps` (See above)
3. Connect to the container via `docker exec -it <container_id> bash`, here you can use directly 'hadoop' instead of the container ID as defined in the docker-compose.yml
   - You should see output like this
   ```
   (Hadoop-Project) abdelmalak@abdelmalak:~/Teaching/SoSe25/PycharmProjects/Hadoop_example$ docker exec -it hadoop bash
    root@eeace327577b:/#
   ```
4. You can now execute commands within the running docker container
5. In the docker-compose.yml, we have the following code to make sure that your local code directory is automatically copied to the docker image : 
    ```
    volumes:
      - ./hadoop_code:/opt/hadoop/code
   ```
6. Check if all files are synchronized via `ls`
   - You should see at least
   ```
   root@eeace327577b:/# cd opt
   root@eeace327577b:/opt# cd hadoop/
   root@eeace327577b:/opt/hadoop# cd code
   root@eeace327577b:/opt/hadoop/code# ls
   inpt.txt  mapper.py reducer.py  run.sh
   ```

## Writing the Hadoop program
- Without modifications the docker container can only host one docker program at a time.
So you are required to copy & paste this folder for each exercise and repeat the previous steps for each exercise folder
- Now to run the code you have **2 main options** :

     1- Run it directly from the local hadoop_example directory which means it should work directly with the current version of run.sh. 
    
     2- Run it from inside the container in which case, you should copy the contents from run_fom_docker.sh to the run.sh file **before** starting up hadoop.
- NOTE : In some cases, especially in windows the first approach can produce some issues in which you make sure to stop running docker, adjust the run.sh according to option 2 then start up hadoop again.
- If your programming is running correctly, you should see an output similar to the following : 
```
     
          [...]
                          WRONG_LENGTH=0
                        WRONG_MAP=0
                        WRONG_REDUCE=0
                File Input Format Counters 
                        Bytes Read=28
                File Output Format Counters 
                        Bytes Written=32
            2025-05-14 11:54:49,471 INFO streaming.StreamJob: Output directory: /output
            MapReduce job complete. Output saved to hadoop_code/output.txt
```
    
- You can check the output of the job on the hadoop_code/output.txt file which should contain : 
```
            20  1
            house	2
            is	1
            no	1
            number  1
```
