OK_FORMAT = True

test = {   'name': 'Q2.b.ii Gradients of the Functions - f2',
    'points': 1,
    'suites': [   {   'cases': [   {   'code': '>>> B = np.array([[4, -2], [-2, 4]])\n'
                                               '>>> a = np.array([[0], [1]])\n'
                                               '>>> b = np.array([[-2], [1]])\n'
                                               '>>> dummy_input = np.array([[2.5], [3.5]])\n'
                                               '>>> output = f2_grad_exact(dummy_input)\n'
                                               '>>> \n'
                                               '>>> target_output = np.array([[1.1858, 5.1032]])\n'
                                               '>>> np.isclose(output, target_output, atol=1e-3).all()\n'
                                               'True',
                                       'failure_message': 'Exact Gradients of f2 Test Failed',
                                       'hidden': False,
                                       'locked': False,
                                       'points': 0,
                                       'success_message': 'Exact Gradients of f2 Test Passed'}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
