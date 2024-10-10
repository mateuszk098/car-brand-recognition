FROM python:3.12.7-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 

WORKDIR /brands

RUN useradd -u 1000 -d /brands -s /bin/bash user && \
    chown -R user:user /brands

COPY --chown=user:user ./build/app/requirements.txt /brands/requirements.txt

RUN --mount=type=bind,source=setup.py,target=/brands/setup.py \
    pip3 install --no-cache-dir -r requirements.txt && \
    rm -vrf ./build ./dist ./*.egg-info

COPY --chown=user:user ./resnet /brands/resnet
COPY --chown=user:user ./app /brands/app

USER user:user
