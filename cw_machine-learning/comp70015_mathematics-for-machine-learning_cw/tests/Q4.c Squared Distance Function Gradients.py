OK_FORMAT = True

test = {   'name': 'Q4.c Squared Distance Function Gradients',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> p = np.random.rand(4)\n>>> q = np.random.rand(4)\n>>> x = np.random.rand(4)\n>>> (4,) == sq_dist_grad(p,q,x).shape \nTrue',
                                       'failure_message': 'Squared Distance Gradient Function Shape Test Failed',
                                       'hidden': False,
                                       'locked': False,
                                       'points': 0,
                                       'success_message': 'Squared Distance Gradient Function Shape Test Passed'},
                                   {   'code': '>>> #! think I prefer to test for us of "grad" from autograd, rather than do memory test\n'
                                               '>>> # 1. this is more robust ; 2. we can them provide demo of the memory use as proof of utility   \n'
                                               '>>> # """ # BEGIN TEST CONFIG\n'
                                               '>>> # name: test_sq_dist_grad_memory\n'
                                               '>>> # points: 2\n'
                                               '>>> # hidden: true \n'
                                               ">>> # success_message: 'Squared Distance Gradient Memory Test Passed'\n"
                                               ">>> # failure_message: 'Squared Distance Gradient Memory Test Failed'\n"
                                               '>>> # """ # END TEST CONFIG\n'
                                               '>>> # %%capture result\n'
                                               '>>> # %load_ext memory_profiler\n'
                                               '>>> # a,b,x = np.random.randn(100), np.random.randn(100), np.random.randn(100)\n'
                                               '>>> # %memit manual_grad_vec(a,b,x)\n'
                                               '>>> # %%capture result2\n'
                                               '>>> # %memit grad(sq_dist_fwd, 2)(a,b,x) \n'
                                               ">>> # get_mem_usage = lambda x: float(str(x)[12:str(x).find('MiB')])\n"
                                               '>>> # 3*get_mem_usage(result) < get_mem_usage(result2)\n',
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
