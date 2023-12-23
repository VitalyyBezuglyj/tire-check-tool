# 3d Homework report

<!-- 
Форма текстового отчета:
- Ваша системная конфигурация
    - OS и версия
    - Модель CPU
    - Количество vCPU и RAM при котором собирались метрики
- Описание решаемой задачи
- Описание структуры вашего model_repository (в формате “$ tree”)
- Секция с метриками по throughput и latency которые вы замерили до всех оптимизаций и после всех оптимизаций
- Объяснение мотивации выбора или не выбора той или иной оптимизации -->

## System configuration

| OS | CPU | vCPU | RAM |
| --- | --- | --- | --- |
| Fedora Linux 39 (WE) | Intel(R) Core(TM)  i7-10510U @  4.90 GHz | 8 | 15.4 GiB |

## Task description

Проверка состояния шин автомобиля по фотографии. По фото с близкого расстояния моедель оценивает износ шин и выдает степень пригоность для эксплуатации.

Пример использования пакета: ![image](img/tire-check-tool_demo.png)

Использованный датасет: [Tire-Check](https://www.kaggle.com/datasets/warcoder/tyre-quality-classification)

Примеры изображений: ![image](img/tires_dataset_demo.png)

## Model repository structure

```bash
triton/model_repository
└── tire_quality_classifier
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```

## Metrics

Для замера метрик импользовалась команда:

```bash
perf_analyzer -m tire_quality_classifier --percentile=95 -u localhost:8500 --shape=IMAGES:1,3,150,150 --concurrency-range 1:8
```

(как базовый шаблон, с соответствующими изменениями для конкретных оптимизаций)

| Optimization | Conq Range |**Throughput, fps** | **Latency, ms** | Details |
| --- | --- | --- | --- | --- |
| Base | $1$ | $75.7842$ | $64.4$ | |
| Base | $2$ | $93.3279$ | $72.4$ | |
| Base | $4$ | $99.33$ | $82.5$ | |
| Base | $8$ | $108.356$ | $98.1$ | |
|  |  |  |  |
| Dynamic Batching | $1$ | $78.8$ | $63.9$ | wo params |
| Dynamic Batching | $2$ | $142.0$ | $62.5$ | wo params |
| Dynamic Batching | $4$ | $183.0$ | $68.8$ | wo params |
| Dynamic Batching | $8$ | $186.7$ | $76.5$ | wo params |
|  |  |  |  |
| Dynamic Batching | $1$ | $75.3$ | $63.8$ | preffered bs $4$ |
| Dynamic Batching | $2$ | $95.5$ | $72.0$ | preffered bs $4$ |
| Dynamic Batching | $4$ | $104.0$ | $81.8$ | preffered bs $4$ |
| Dynamic Batching | $8$ | $110.8$ | $96.6$ | preffered bs $4$ |
|  |  |  |  |
| Dynamic Batching | 1 | $77.9$ | $64.0$ | max delay $100$ |
| Dynamic Batching | 2 | $93.4$ | $72.2$ | max delay $100$ |
| Dynamic Batching | 4 | $109.1$ | $82.2$ | max delay $100$ |
| Dynamic Batching | 8 | $112.3$ | $96.7$ | max delay $100$ |
|  |  |  |  |
| Dynamic Batching | 1 | $76.5$ | $63.9$ | max delay $200$ |
| Dynamic Batching | 2 | $92.9$ | $72.0$ | max delay $200$ |
| Dynamic Batching | 4 | $102.3$ | $82.4$ | max delay $200$ |
| Dynamic Batching | 8 | $214.5$ | $79.0$ | max delay $200$ |
|  |  |  |  |
| More instances | 1 | $76.1$ | $63.5$ | 2, with dynamic batching |
| More instances | 2 | $94.6$ | $71.7$ | 2, with dynamic batching |
| More instances | 4 | $108.6$ | $87.6$ | 2, with dynamic batching |
| More instances | 8 | $115.7$ | $96.4$ | 2, with dynamic batching |
|  |  |  |  |
| More instances | 1 | $72.5$ | $64.0$ | 4, with dynamic batching |
| More instances | 2 | $98.8$ | $71.8$ | 4, with dynamic batching |
| More instances | 4 | $107.9$ | $81.4$ | 4, with dynamic batching |
| More instances | 8 | $117.0$ | $95.1$ | 4, with dynamic batching |
|  |  |  |  |
| Open Vino | 1 | $73.4$ | $63.7$ | with dynamic batching, 4 instances |
| Open Vino | 2 | $95.8$ | $71.9$ | with dynamic batching, 4 instances |
| Open Vino | 4 | $106.5$ | $88.4$ | with dynamic batching, 4 instances |
| Open Vino | 8 | $119.4$ | $94.9$ | with dynamic batching, 4 instances |

## Optimization details

### Dynamic Batching

Dynamic batching - позволяет автоматически увеличивать размер батча, если в очереди накопилось достаточно запросов, что позволяет увеличить throughput и уменьшить latency.

Эта оптимизация может использоваться и без параметров, но позволяет указать некоторые для повышения эффективности.

Были проведены жксперименты с такими параметрами как:

- `preferred_batch_size`: предпочтительный размер батча, который будет использоваться, если в очереди накопилось достаточно запросов

- `max_queue_delay_microseconds`: максимальное время ожидания запросов в очереди, после которого они будут отправлены на выполнение

Такжже можно было указать queue_policy или priority_levels, но это предполагает слишком большое количество экспериментов, и с учётом игрушечности задачи и использования ноутбука для экспериментов, выглядит нецелесообразным.

### More instances

Позволяет запускать несколько экземпляров модели, что позволяет увеличить throughput и уменьшить latency.

Собственно, попробовал 2 и 4 экземпляра, больше некуда тк ноутбук)

### Open Vino

Позволяет использовать модель, оптимизированную для работы на CPU с помощью OpenVino.

### Conq Range

Позволяет указать диапазон количества запросов, для которых будет использоваться один экземпляр модели.

### Notes

- По-хорошему следовало провести эксперименты с разными оптимизациями по одновременно, но это предполагает слишком большое количество экспериментов

- Max batch size - $4$ - был выбран из головы~


## Conclusion

- В целом, все оптимизации позволяют увеличить throughput и уменьшить latency, но не в разы, а в пределах $10-20\%$

- Наилучший результат показала оптимизация с использованием OpenVino, но это неудивительно, тк она предполагает использование модели, оптимизированной для работы на CPU, но прям чуть-чуть

- Итоговая версия конфига: 
- - `preferred_batch_size`: $4$
- - `max_queue_delay_microseconds`: $200$
- - `instance_group`: $4$
- - `inference_provider`: `openvino_cpu`








