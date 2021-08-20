import pytest

from muttlib.dbconn.snowflake import SnowflakeClient, read_private_key


@pytest.fixture
def private_key():
    return b"-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\nDEK-Info: AES-256-CBC,D91B50B98E835D5159B0D5113302B6E9\n\nee1J2IETqqdp38i6sxv6DTQ3hd6AqwPsQY3+pH5yAhFEFskXjgjztnhtMxNioYdW\nrHgMHV2UtLOHmF38rcbNhsZQvSWlJIx+grqzWylyOtv46eLSTu6Qaa+jX97+VY+r\ne+lkux9OADzci9fzpRqu3LYJSf86K0X46obXuusb1NfW+aWKHENcGgi3N/ryyhb5\n2t6TsNtkhGuK2uPCj5sfDB0wZoXhv+JEpF3BkxdWlZ+J+SKT8NfvKYUYCG9lGsdl\nMmh5V1c3VU3nDzy7aUwy8Igzs9pKnWWjoayWWltvzoK8bHLaT9rlRB5MrMucRoVi\noYaGmXd1G0+jv83+9f0X7teMFaNFIc3sDhts6CUCDB/WssShHfKV7RIVYZoqcId7\nixQ+xJ9DAHnjP9jg2cmUoBW0ReKt19qOyvx+iYbu1qD4cWNq7bqrB4UNGyvdV1E1\n-----END RSA PRIVATE KEY-----\n"


@pytest.fixture
def pk_passphrase():
    return "passphrase"


@pytest.fixture
def db_creds():
    return {
        "username": "dummy-user",
        "account": "dummy-account",
        "warehouse": "dummy-warehouse",
        "database": "dummy-databse",
        "schema": "dummy-schema",
        "password": "dummy-password",
    }


def test_conn_str_property(db_creds):
    username = db_creds["username"]
    password = db_creds["password"]
    account = db_creds["account"]
    database = db_creds["database"]
    schema = db_creds["schema"]
    warehouse = db_creds["warehouse"]
    sf_client = SnowflakeClient(
        username=username,
        password=password,
        account=account,
        database=database,
        schema=schema,
        warehouse=warehouse,
    )
    db_uri = f"snowflake://{username}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"
    assert sf_client.conn_str == db_uri


def test_read_private_key(private_key, pk_passphrase):
    pk = read_private_key(
        private_key,
        pk_passphrase,
        encoding_type="PEM",
        format_private=True,
        format_type="TraditionalOpenSSL",
        encryption=False,
    )
    assert (
        pk
        == b"-----BEGIN RSA PRIVATE KEY-----\nMIIBPAIBAAJBAMQSAUK/x8JJMvhX3+XEj41Su3NF02/EEH6H6eow/x/LZx+NL+EE\n/u+0e2NW0ZuK3dp8K3s7y9Julnv9Q5VGbf0CAwEAAQJBAJWNNjD3nyJuOtZ6EGlt\nWCFvbVMre27QmdQpTx42aSKR6Tny6bdxtFpPlXlnOxCkrsMHh3UYnfB71Vsd1KGW\n+vECIQDhgqtPQKowO09PSXQK2uwmVVPehm35D8il7a4rvEzNgwIhAN6UXCIViAoH\nZoFAec8sIEl6BbrViyuZCMUc3h7BAn5/AiAvXyqraFMX9K2RY0W8LgbjepM2sJiT\ndExbBtXKnDCqDwIhAIrFPNwTSInYK1SSel9sR4UICuJ9mRNJimo6oVHTTFbJAiEA\nmH5ElHkDDpB9ISsZigiZjip/rPIeS2lQ13qGxqgENzw=\n-----END RSA PRIVATE KEY-----\n"
    )


def test_except_invalid_encoding_type(private_key, pk_passphrase):
    with pytest.raises(AttributeError):
        pk = read_private_key(
            private_key,
            pk_passphrase,
            encoding_type="wrong-encoding-type",
            format_private=True,
            format_type="TraditionalOpenSSL",
            encryption=False,
        )


def test_except_invalid_private_key_format(private_key, pk_passphrase):
    with pytest.raises(AttributeError):
        pk = read_private_key(
            private_key,
            pk_passphrase,
            encoding_type="PEM",
            format_private=True,
            format_type="wrong-format-type",
            encryption=False,
        )


def test_except_failed_load_private_key(private_key):
    with pytest.raises(Exception):
        pk = read_private_key(
            private_key,
            "wrong-passphrase",
            encoding_type="PEM",
            format_private=True,
            format_type="TraditionalOpenSSL",
            encryption=False,
        )
