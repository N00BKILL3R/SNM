from models.partition import Budget, select_partition_point


def test_partition_auto():
    cfg = {"latent_dim": 128}
    p = select_partition_point(cfg, Budget(60, 32, 0.4))
    assert p in {"p0", "p1", "p2"}
