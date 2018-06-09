FROM golang

RUN mkdir /app 
RUN go get -u github.com/golang/dep/cmd/dep
ADD . /go/src/github.com/delaneyj/cogent
WORKDIR /go/src/github.com/delaneyj/cogent
RUN dep init
CMD go test -tags=avx -run ^Test_Flowers$