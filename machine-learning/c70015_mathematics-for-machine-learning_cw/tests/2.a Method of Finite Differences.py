OK_FORMAT = True

test = {   'name': '2.a Method of Finite Differences',
    'points': 4,
    'suites': [   {   'cases': [   {   'code': '>>> B = np.array([[4, -2], [-2, 4]])\n'
                                               '>>> a = np.array([[0], [1]])\n'
                                               '>>> b = np.array([[-2], [1]])\n'
                                               '>>> def f1(x):\n'
                                               '...     return float(x.T @ B @ x - x.T @ x + a.T @ x - b.T @ x)\n'
                                               '>>> dummy_input = np.array([[2.5], [3.5]])\n'
                                               '>>> output = grad_fd(f1, dummy_input)\n'
                                               '>>> \n'
                                               '>>> target_output = np.array([[3.00003, 11.00003]])\n'
                                               '>>> np.isclose(output, target_output, atol=1e-3).all()\n'
                                               'True',
                                       'failure_message': 'Finite Differences on f1 Test Failed',
                                       'hidden': False,
                                       'locked': False,
                                       'points': 1,
                                       'success_message': 'Finite Differences on f1 Test Passed'},
                                   {   'code': '>>> B = np.array([[4, -2], [-2, 4]])\n'
                                               '>>> a = np.array([[0], [1]])\n'
                                               '>>> b = np.array([[-2], [1]])\n'
                                               '>>> def f2(x):\n'
                                               '...     return float(np.cos((x - b).T @ (x - b)) + (x - a).T @ B @ (x - a))\n'
                                               '>>> dummy_input = np.array([[2.5], [3.5]])\n'
                                               '>>> output = grad_fd(f2, dummy_input)\n'
                                               '>>> \n'
                                               '>>> target_output = np.array([[1.18572957, 5.10321673]])\n'
                                               '>>> np.isclose(output, target_output, atol=1e-3).all()\n'
                                               'True',
                                       'failure_message': 'Finite Differences on f2 Test Failed',
                                       'hidden': False,
                                       'locked': False,
                                       'points': 0,
                                       'success_message': 'Finite Differences on f2 Test Passed'}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
