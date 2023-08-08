# Fuzzy C-Means + mpmath

*Projeto desenvolvido como Trabalho de Conclusão do Curso de Anderson de Alencar Barros*

Este projeto é a implementação do método Fuzzy C-Means usando pontos flutuantes de precisão arbitrária, conhecido como *bigfloats*, sendo útil para projetos que requerem alta precisão onde um ponto de fluante de precisão dupla poder ocasionar erros de divisão por zero ou não ser preciso o suficiente.

Para explorar o projeto comece fazendo um clone do projeto,

```bash
git clone https://github.com/AndersonAlencarBarros/fcm
```

Os requisitos para executar esse projeto são,

-   Poetry
-   Python 3.10

Para instalar use o comando,

```bash
pip install fcm-mpmath
```

## Exemplo


```python
from fcm_mpmath import FCM
import numpy as np


X = np.array(
    [
        [1, 3],
        [2, 5],
        [4, 8],
        [7, 9],
    ]
)


fcm = FCM(n_clusters=2, mu=2)

fcm.fit(data=X)

print(fcm.centers) 
```
 


## Referências


```
BEZDEK, J. C.; EHRLICH, R.; FULL, W. Fcm: The fuzzy c-means clustering algorithm.
Computers & geosciences, Elsevier, v. 10, n. 2-3, p. 191–203, 1984. 
```
