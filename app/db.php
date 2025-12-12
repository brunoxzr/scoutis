<?php
$pdo = new PDO(
  "pgsql:host=127.0.0.1;port=5432;dbname=scoutis",
  "postgres",
  "senha123",
  [
    PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
    PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC
  ]
);
