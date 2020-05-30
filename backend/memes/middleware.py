def allow_options_requests_wrapper(get_response):
    # One-time configuration and initialization.

    def allow_options_requests(request):
        # Code to be executed for each request before
        # the view (and later middleware) are called.

        response = get_response(request)

        # Code to be executed for each request/response
        # after the view is called.
        # 1. Add cors headers:
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Headers"] = "Content-Type"

        # 2. allow options requests on our endpoint
        if (response.status_code == 405 and
                request.method == "OPTIONS"):
            response.status_code = 200
            response.content = ""

        return response

    return allow_options_requests
