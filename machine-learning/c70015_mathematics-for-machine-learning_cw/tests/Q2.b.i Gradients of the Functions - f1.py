OK_FORMAT = True

test = {   'name': 'Q2.b.i Gradients of the Functions - f1',
    'points': 1,
    'suites': [   {   'cases': [   {   'code': '>>> B = np.array([[4, -2], [-2, 4]])\n'
                                               '>>> a = np.array([[0], [1]])\n'
                                               '>>> b = np.array([[-2], [1]])\n'
                                               '>>> dummy_input = np.array([[2.5], [3.5]])\n'
                                               '>>> output = f1_grad_exact(dummy_input)\n'
                                               '>>> \n'
                                               '>>> target_output = np.array([[3.0, 11.0]])\n'
                                               '>>> np.isclose(output, target_output, atol=1e-3).all()\n'
                                               'True',
                                       'failure_message': 'Exact Gradients of f1 Test Failed',
                                       'hidden': False,
                                       'locked': False,
                                       'points': 0,
                                       'success_message': 'Exact Gradients of f1 Test Passed'}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
