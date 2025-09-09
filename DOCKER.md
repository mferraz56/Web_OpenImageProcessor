# Docker - Instruções de Uso

## Construir a imagem Docker

Para construir a imagem Docker da aplicação, execute:

```bash
docker build -t image-processor .
```

## Executar o container

Para executar o container da aplicação:

```bash
docker run -p 7000:7000 -v $(pwd)/Entrada:/app/Entrada -v $(pwd)/Saida:/app/Saida image-processor
```

### No Windows (PowerShell):
```powershell
docker run -p 7000:7000 -v ${PWD}/Entrada:/app/Entrada -v ${PWD}/Saida:/app/Saida image-processor
```

## Opções explicadas

- `-p 7000:7000`: Mapeia a porta 7000 do container para a porta 7000 do host
- `-v $(pwd)/Entrada:/app/Entrada`: Monta o diretório local "Entrada" no container para upload de imagens
- `-v $(pwd)/Saida:/app/Saida`: Monta o diretório local "Saida" no container para salvar imagens processadas
- `image-processor`: Nome da imagem Docker

## Acessar a aplicação

Após executar o container, acesse a aplicação em:
http://localhost:7000

## Executar em modo detached (background)

Para executar o container em background:

```bash
docker run -d -p 7000:7000 -v $(pwd)/Entrada:/app/Entrada -v $(pwd)/Saida:/app/Saida --name image-processor-app image-processor
```

## Parar o container

```bash
docker stop image-processor-app
```

## Remover o container

```bash
docker rm image-processor-app
```

## Visualizar logs do container

```bash
docker logs image-processor-app
```

## Verificar status de saúde (healthcheck)

O container inclui um healthcheck que verifica se a aplicação está respondendo:

```bash
docker ps
```

O status aparecerá como:
- `healthy` - Aplicação funcionando corretamente
- `unhealthy` - Aplicação com problemas
- `starting` - Container iniciando (período de grace)

Para verificar detalhes do healthcheck:

```bash
docker inspect image-processor-app --format='{{.State.Health.Status}}'
```
