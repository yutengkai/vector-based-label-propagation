from vlp.utils.memory import dense_feasible

def test_memory_guard():
    assert dense_feasible(1000)  # small
    assert not dense_feasible(100000)  # huge
