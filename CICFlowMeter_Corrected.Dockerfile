FROM maven:3.8.1-openjdk-8

ARG DEBIAN_FRONTEND=noninteractive

# Update sources.list to use archive repositories for buster
RUN echo "deb http://archive.debian.org/debian/ buster main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security buster/updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "Acquire::Check-Valid-Until false;" > /etc/apt/apt.conf.d/99no-check-valid-until && \
    apt-get update && \
    apt-get install -y git parallel libpcap-dev wget unzip && \
    rm -rf /var/lib/apt/lists/*

# Install Gradle manually
RUN wget https://services.gradle.org/distributions/gradle-6.9.1-bin.zip -P /tmp && \
    unzip -d /opt/gradle /tmp/gradle-6.9.1-bin.zip && \
    ln -s /opt/gradle/gradle-6.9.1/bin/gradle /usr/bin/gradle

RUN git clone https://github.com/ahlashkari/CICFlowMeter.git

WORKDIR /CICFlowMeter/jnetpcap/linux/jnetpcap-1.4.r1425
RUN mvn install:install-file -Dfile=jnetpcap.jar -DgroupId=org.jnetpcap -DartifactId=jnetpcap -Dversion=1.4.1 -Dpackaging=jar

WORKDIR /CICFlowMeter

# Use the correct task name: fatJar instead of fatJarCMD
RUN if [ -f ./gradlew ]; then chmod +x ./gradlew && ./gradlew fatJar; else gradle fatJar; fi

ENV LD_LIBRARY_PATH=/CICFlowMeter/jnetpcap/linux/jnetpcap-1.4.r1425/

ENTRYPOINT ["java", "-Djava.library.path=/CICFlowMeter/jnetpcap/linux/jnetpcap-1.4.r1425/", "-jar", "build/libs/CICFlowMeter-4.0.jar"]
