# Submission: stariy bog

Статус: все 10 автоматических проверок на публичных данных T4 пройдены.
Оформление сдачи и перенос в репозиторий команды ещё не завершены.

## Ссылки

- Репозиторий: https://github.com/BAITC-Hacks/hack-8db9951c-stariy-bog
- Репозиторий разработки: https://github.com/samenov01/soup-kernel-lab
- Ветка кандидата: https://github.com/samenov01/soup-kernel-lab/tree/speed50-memory5g
- Репозиторий команды восстановлен до исходного README. Для сдачи проверенный код
  нужно отдельно перенести туда; личный репозиторий не заменяет конкурсный.
- Проверенный коммит разработки: https://github.com/samenov01/soup-kernel-lab/commit/33641b98b154e95cb14849c6f3264aee23eefd78
- Финальный коммит в репозитории команды: ещё не создан.
- Kaggle: https://www.kaggle.com/code/diassamenov/notebookbbefaa73fb
- Автоматический проверенный запуск: https://www.kaggle.com/code/diassamenov/soup-kernel-lab-auto
  Версия 19, приватная; публичная финальная версия для сдачи ещё не оформлена.
- Видео до 3 минут: не предоставлено.
- Презентация: не предоставлена.

## Команда

Название `stariy bog` взято из README репозитория команды.
Имена, роли, аккаунты участников и контакт капитана не предоставлены.

## Собственные кернелы

Vocabulary-кернелы находятся в `solution/vocab.py`, NF4 — в `solution/nf4.py`,
RMSNorm — в `solution/rmsnorm.py`, RoPE — в `solution/rope.py`,
projection IO — в `solution/projection_io.py`, LoRA — в `solution/compact_lora.py`,
attention IO — в `solution/attention_io.py`.
Общая доля: 14.8% в одном полном повторе,
минимум выше требуемых 5%.

| Кернел | Работа |
|---|---|
| `vocab_partial_kernel` | Блочные max, сумма exp, момент для entropy |
| `vocab_ce_finish_kernel` | Объединение блоков, CE и logsumexp |
| `vocab_ce_backward_kernel` | Градиент logits с учётом маски и upstream gradient |
| `vocab_entropy_finish_kernel` | Энтропия из блочных статистик |
| `nf4_unpack_scale_kernel` | Распаковка и масштабирование NF4-весов |
| `rmsnorm_forward_kernel` | Frozen fp32 RMSNorm forward |
| `rmsnorm_backward_kernel` | Градиент по входу RMSNorm |
| `rope_transform_kernel` | RoPE forward и транспонированное преобразование градиента |
| `projection_gather_kernel` | Gather действительных позиций с преобразованием dtype |
| `projection_scatter_kernel` | Scatter и заполнение padding нулями |
| `sparse_lora_add_kernel` | Half scaling LoRA, scatter и сложение с base |
| `sparse_lora_gradient_kernel` | Gather градиента и scaling с исходным округлением |
| `attention_pack_kernel` | Прямой BHLD gather/cast для native variable-length attention |

Пример SwiGLU организаторов сохранён и не заявляется собственным.
Источники математических идей и API приведены в README; код кернелов написан заново.

## Результаты Kaggle T4

| Вариант | Полное время, с | Пик, MiB | Eval loss |
|---|---:|---:|---:|
| Base | 85.84 | 14345 | 2.04970 |
| Собственное решение | 64.73 | 3989 | 2.04952 |

Один повтор; требуется дополнительное повторение. Memory ratio 3.5961x,
speedup 1.3261x, изменение eval loss -0.009%.
Измерения и ограничения: `docs/T4_RUN_08.md`.

- [x] Все 10 проверок пройдены для проверенного коммита на публичном T4-запуске.
- [x] README описывает реализацию, воспроизведение и ограничения.
- [ ] Публичная финальная версия Kaggle привязана к опубликованному коммиту.
- [ ] Видео и презентация добавлены.

## Ограничения и содержание презентации

Запас доли над порогом составляет около 9.8 процентного пункта: скрытые данные
могут изменить допуск. Память снизилась с 14345 до 3989 MiB.
Цель <5000 MiB выполнена, ≤50 с по времени не достигнута. Результат относится
ко всей совокупности патчей; изолированного замера каждого компонента нет.
Внутренние активации decoder хранятся в FP16 при FP32-статистиках норм;
веса и trainer settings сохранены. Округление отличается от прежнего пути,
качество и кривая проверены, побитовая тождественность не заявляется.
Native variable-length attention отделяет строки cumulative offsets;
логический batch сохранён. Frozen MLP пересчитывает gate/up в backward,
экономя память ценой вычислений. Изолированный выигрыш каждого патча не заявляется.
Unit-тесты: 111 passed, 14 warnings in 65.18s. Нужны публичная сохранённая
версия Kaggle, перенос в репозиторий команды, видео и презентация.
Решение жюри по коду и скрытым данным пока отсутствует.
