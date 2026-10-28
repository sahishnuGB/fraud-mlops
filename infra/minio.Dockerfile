# MinIO stopped publishing community container images in late 2025, so build a pinned
# release from source (AGPLv3). The first `docker compose up` takes a few minutes.
FROM golang:1.25-alpine AS build
ARG MINIO_RELEASE=RELEASE.2025-10-15T17-29-55Z
RUN apk add --no-cache git \
 && CGO_ENABLED=0 GOTOOLCHAIN=auto go install "github.com/minio/minio@${MINIO_RELEASE}"

FROM alpine:3.22
RUN apk add --no-cache ca-certificates curl
COPY --from=build /go/bin/minio /usr/local/bin/minio
EXPOSE 9000 9001
ENTRYPOINT ["minio"]
