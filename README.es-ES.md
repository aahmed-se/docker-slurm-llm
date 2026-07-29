

# Entrenamiento Distribuido con PyTorch y Slurm

Un ejemplo mínimo y educativo que demuestra el aprendizaje profundo distribuido utilizando PyTorch y Slurm en un clúster multi-nodo basado en Docker.

## Visión General

Este proyecto muestra el **Paralelismo de Tensores** a través de múltiples nodos de computación utilizando:

- **PyTorch Distributed** para entrenamiento multi-proceso
- **Slurm** para gestión de carga de trabajo y programación de tareas
- **Docker Compose** para simular un clúster multi-nodo localmente

Está diseñado para aprender conceptos de entrenamiento distribuido sin acceso a un clúster HPC real.

## Características

- **Clúster Slurm de 2 Nodos** en ejecución dentro de contenedores Docker
- **Entrenamiento con Paralelismo de Tensores** con particionamiento del modelo entre nodos
- **Distribución Automática de Nodos** utilizando las restricciones de recursos de Slurm
- **Entrenamiento Sincronizado** con sincronización de gradientes vía `all_reduce`
- **Modelo de Lenguaje a Nivel de Caracteres** entrenado con texto de Shakespeare

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                      Slurm Controller                       │
│                         (node-1)                            │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
        ┌───────▼──────┐            ┌───────▼──────┐
        │   Worker 1   │            │   Worker 2   │
        │   (node-1)   │            │   (node-2)   │
        │   Rank 0     │◄──────────►│   Rank 1     │
        │              │  all_reduce│              │
        └──────────────┘            └──────────────┘
               ▲                           ▲
               │                           │
         Model Shard 1                Model Shard 2
         (32 features)                (32 features)
```

## Inicio Rápido

### Prerrequisitos

- Docker y Docker Compose
- Comprensión básica de conceptos de entrenamiento distribuido

### 1. Clonar y Construir

```bash
git clone https://github.com/aahmed-se/docker-slurm-llm
cd docker-slurm-llm
docker compose build
```

### 2. Iniciar el Clúster

```bash
docker compose up -d
```

Verifica que ambos nodos estén en ejecución:
```bash
docker exec node-1 sinfo -N -l
```

Salida esperada:
```
NODELIST   NODES PARTITION       STATE CPUS
node-1         1    debug*        idle    1
node-2         1    debug*        idle    1
```

### 3. Preparar los Datos de Entrenamiento

```bash
docker exec -it node-1 python3 /data/prepare_data.py
```

Esto descarga el conjunto de datos Tiny Shakespeare y construye el vocabulario.

### 4. Enviar el Trabajo de Entrenamiento

```bash
docker exec -it node-1 bash -c "cd /data && sbatch job.sh"
```

Monitorea el trabajo:
```bash
docker exec node-1 squeue
tail -f workspace/training_job.log
```

### 5. Verificar la Distribución

Revisa el archivo de registro en busca de evidencia de ejecución multi-nodo:

```bash
grep "SLURMD_NODENAME" workspace/training_job.log
```

Deberías ver:
```
[Rank: 0 | SLURMD_NODENAME: node-1 | ...]
[Rank: 1 | SLURMD_NODENAME: node-2 | ...]
```

### 6. Limpieza

```bash
docker compose down
```

## Estructura del Proyecto

```
.
├── Dockerfile                 # Container definition with Slurm + PyTorch
├── docker-compose.yml         # Multi-node cluster orchestration
├── configs/
│   ├── slurm.conf            # Slurm cluster configuration
│   └── cgroup.conf           # Resource control configuration
├── scripts/
│   └── entrypoint.sh         # Container initialization script
└── workspace/
    ├── prepare_data.py       # Dataset download and preprocessing
    ├── train.sh              # Slurm job submission script
    ├── train_tp.py           # Distributed training implementation
    └── training_job.log      # Training output (generated)
```

## Cómo Funciona

### Configuración de Slurm

La clave para garantizar la distribución multi-nodo:

```bash
# slurm.conf
NodeName=node-1 NodeAddr=node-1 CPUs=1 State=UNKNOWN
NodeName=node-2 NodeAddr=node-2 CPUs=1 State=UNKNOWN

SelectType=select/cons_tres
SelectTypeParameters=CR_Core
```

Con `CPUs=1` por nodo y `--cpus-per-task=1`, Slurm **debe** distribuir las tareas en ambos nodos.

### Script del Trabajo de Entrenamiento

```bash
#!/bin/bash
#SBATCH --nodes=2             # Use 2 nodes
#SBATCH --ntasks=2            # Launch 2 processes
#SBATCH --ntasks-per-node=1   # 1 process per node
#SBATCH --cpus-per-task=1     # 1 CPU per process

srun --nodelist=node-1,node-2 python3 /data/train_tp.py
```

### Implementación del Paralelismo de Tensores

El modelo se particiona entre nodos:

```python
class TPLinear(nn.Module):
    def __init__(self, input_dim, output_dim, world_size):
        super().__init__()
        # Each node gets a shard of the output dimension
        self.partition_dim = output_dim // world_size
        self.weight = nn.Parameter(torch.randn(input_dim, self.partition_dim))
    
    def forward(self, x):
        return torch.matmul(x, self.weight)
```

Los gradientes se sincronizan usando `all_reduce`:

```python
reduced_loss = loss.clone().detach()
dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
reduced_loss /= world_size
```

## Conceptos Clave Demostrados

### 1. **Inicialización del Grupo de Procesos**
```python
dist.init_process_group('gloo', rank=rank, world_size=world_size)
```

### 2. **Particionamiento del Modelo**
Cada rango posee una partición de los parámetros del modelo (32 de 64 características de salida).

### 3. **Comunicación Colectiva**
Todos los rangos sincronizan los valores de pérdida usando `all_reduce` para un entrenamiento consistente.

### 4. **SPMD (Single Program, Multiple Data)**
El mismo código se ejecuta en todos los nodos, pero opera en diferentes particiones del modelo.

## Solución de Problemas

### ¿Ambos procesos en el mismo nodo?

Verifica la distribución real:
```bash
docker exec node-1 grep "SLURMD_NODENAME" /data/training_job.log
```

Si ambos muestran `node-1`, asegúrate de que:
1. La configuración de Slurm tenga `CPUs=1` por nodo
2. El script del trabajo use `--cpus-per-task=1`
3. El clúster se haya reiniciado después de los cambios en la configuración

### ¿Errores de autenticación de Munge?

El Dockerfile genera una clave munge aleatoria. Si los nodos no pueden comunicarse:
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### ¿Trabajo atascado en la cola?

Verifica el estado de los nodos:
```bash
docker exec node-1 scontrol show nodes
```

Ambos nodos deberían mostrar `State=IDLE`.

## Escalado

Para agregar más nodos, modifica `docker-compose.yml` y `slurm.conf`:

```yaml
# docker-compose.yml
services:
  node-3:
    build: .
    image: slurm-llm-image
    hostname: node-3
    # ...
```

```bash
# slurm.conf
NodeName=node-3 NodeAddr=node-3 CPUs=1 State=UNKNOWN
PartitionName=debug Nodes=node-[1-3] Default=YES
```

Luego actualiza `train.sh`:
```bash
#SBATCH --nodes=3
#SBATCH --ntasks=3
```

## Recursos de Aprendizaje

- [Visión General de PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Documentación de Slurm](https://slurm.schedmd.com/documentation.html)
- [Paralelismo de Tensores Explicado](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)

## Agradecimientos

- Conjunto de datos: [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) por Andrej Karpathy
- Inspirado en flujos de trabajo de entrenamiento HPC del mundo real
