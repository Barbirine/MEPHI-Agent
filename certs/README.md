# Публичный сертификат для GigaChat

`russian_trusted_root_ca.pem` — публичный корневой сертификат Russian Trusted Root CA. Он не содержит приватного ключа или учётных данных.

Источник: https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt

Адрес приведён в официальной документации: https://developers.sber.ru/docs/ru/gigachat/certificates

Скачан 24 октября 2026 года через HTTPS с включённой проверкой TLS. Данные проверены командой OpenSSL:

- Subject: `C=RU, O=The Ministry of Digital Development and Communications, CN=Russian Trusted Root CA`.
- Действует с 1 марта 2022 года по 27 февраля 2032 года (UTC).
- SHA-256 fingerprint DER: `D2:6D:2D:02:31:B7:C3:9F:92:CC:73:85:12:BA:54:10:35:19:E4:40:5D:68:B5:BD:70:3E:97:88:CA:8E:CF:31`.

Приложение добавляет сертификат к стандартным доверенным CA только в собственном SSL-контексте HTTPX. Проверка имени сервера и сертификата остаётся включённой. Системное хранилище и пакет certifi не изменяются. Для обновлённой цепочки укажите свой доверенный PEM в `GIGACHAT_CA_BUNDLE_FILE`.

Загрузка сертификата и локальная проверка SSL-контекста не являются проверкой соединения с GigaChat API: реальные запросы к нему не выполнялись.
