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

Запуск
```
./target/release/detect_objects --num-workers 2 --log-level info
```

## Визуализация детекции лиц людей

Открыть файл `index.html` из корня проекта и выбрать любое изображение. В случае успеха лицо человека на изображении будет обведено в зеленый прямоугольник.