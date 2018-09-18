# Use Official Microsoft Azure CLI image
FROM continuumio/miniconda3:4.5.4

# Install any needed packages specified in requirements.txt
RUN apt-get update
RUN apt-get install -y autoconf=2.69-10 automake=1:1.15-6 build-essential=12.3 libtool=2.4.6-2 python-dev=2.7.13-2 jq=1.5+dfsg-1.3

# Set the working directory to /
WORKDIR /
# Copy the directory contents into the container at /
COPY . /

RUN make requirements

RUN chmod +x -R /deploy

CMD ["make", "deploy"]


