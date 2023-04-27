import run_tests

def test_1a():
    run_tests.run("predict_gp", "12wmEwKKRlBCc-OGYxirQ6RCoSyymDoEd", 
                  output_dim=9)

def test_1b():
    run_tests.run("ucb", "135AZ46v88dR7bVhJtbiMKzeDmXs05_82", output_dim=2)

def test_2a():
    run_tests.run("sample_gp", "13Ck3tCUzDHo6VJUzQ2RILXUebpKD3Hgg", 
                  output_dim=3)
    
def test_2b():
    run_tests.run("mcei", "13PkR9EpBIIzGmmRUDwQOrIu9tu7DaqFl", output_dim=2)