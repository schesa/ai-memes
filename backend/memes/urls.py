from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from graphene_django.views import GraphQLView

from . import views

urlpatterns = [
    # path("memes", views.memes, name="memes"),
    path('graphqlapi/', csrf_exempt(GraphQLView.as_view(graphiql=True))),
    path('graphql/', csrf_exempt(GraphQLView.as_view(graphiql=False))),
]
