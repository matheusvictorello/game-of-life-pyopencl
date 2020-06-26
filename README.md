# Game Of Life no PyOpenCL

## Setup

Como esse projeto usa OpenCL, é necessário que uma versão (1.2+) do OpenCL esteja instalado no sistema.
Os seguintes pacotes do python são usados:

* pyopencl 
* numpy 
* pygame
* timeit

Podem ser instalados com:
```bash
pip install *nome*
```

## Arquivos

Existem três arquivos nesse repo:
	
### cl.py

É a implementação que usa opencl. Para rodar basta executar o seguinte:

```bash
python cl.py
```

### nocl.py

É a implementação que não usa opencl. Para rodar basta executar o seguinte:

```bash
python nocl.py
```

### info.py

É o código auxiliar para saber o tamanha de grupo de trabalho preferido do device. Para rodar basta executar o seguinte:

```bash
python info.py
```