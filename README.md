# Обнаружение объектов на изображениях

Для решения задачи детекции используется нейронная сеть `YOLO`.

## Зависимости

Приложение использует [ONNX runtime](https://github.com/microsoft/onnxruntime/releases).
Протестирована версия 1.23.2
Для установки достаточно скачать архив и указать путь в .bashrc:
```
export LD_LIBRARY_PATH=$HOME/Install/onnxruntime-linux-x64-1.23.2/lib:$LD_LIBRARY_PATH
```

## Сборка и запуск приложения

Сборка
```
cargo build --release
```

Параметры конфигурирования:
- `num-workers` число воркер тредов (actix workers)
- `log-level` уровень логирования (`debug`, `info`, `error`)
- `model` определяет тип детекции (`face` - детекция лиц, `object` - детекция объектов)

Пример запуска в режиме детекции лиц людей:
```
./target/release/detect_objects --num-workers 2 --log-level info --model face
```

Пример запуска в режиме детекции объектов:
```
./target/release/detect_objects --num-workers 2 --log-level info --model object
```

## Визуализация результатов детекции

Открыть файл `index.html` из корня проекта и выбрать любое изображение.
В случае успеха, объект или лицо человека на изображении будут обведены в зеленый прямоугольник.