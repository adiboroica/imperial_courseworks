OK_FORMAT = True

test = {   'name': 'Q4.d Squared Distance Autodiff',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> our_grad = grad(sq_dist, 0)(z) # This is calling our custom gradient function\n'
                                               '>>> their_grad = grad(sq_dist_fwd, 2)(p,q,z) # This is uses slow gradient primitives\n'
                                               '>>> np.allclose(our_grad, their_grad, atol=1e-3)\n'
                                               'True',
                                       'failure_message': 'Squared Distance Autodiff Value Test Failed',
                                       'hidden': False,
                                       'locked': False,
                                       'points': 2,
                                       'success_message': 'Squared Distance Autodiff Value Test Passed'}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
