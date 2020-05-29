import datetime
from memes.models import Meme
from memes.grpc_client import get_url
import graphene
from graphene import ObjectType
from graphene_django.types import DjangoObjectType


class MemeType(DjangoObjectType):
    class Meta:
        model = Meme


class Query(graphene.ObjectType):
    all_memes = graphene.List(MemeType)
    meme = graphene.Field(
        MemeType,
        id=graphene.Int(),
        caption=graphene.String()
    )

    def resolve_all_memes(self, info, **kwargs):
        return Meme.objects.all()

    def resolve_meme(self, info, **kwargs):
        _id = kwargs.get("id")
        _caption = kwargs.get("caption")
        if _id:
            return Meme.objects.get(id=_id)
        if _caption:
            return Meme.objects.filter(caption=_caption).first()
        return None


class AddMemeInput(graphene.InputObjectType):
    caption = graphene.String()
    templateid = graphene.String()
    #  = graphene.String()
    # dueDate = graphene.String()
    # completed = graphene.Boolean()


class AddMemeMutation(graphene.Mutation):
    class Arguments:
        meme = AddMemeInput(required=True)

    meme = graphene.Field(MemeType)

    def mutate(self, info, meme):
        new_meme = Meme.objects.create(
            caption=meme.caption,
            templateid=meme.templateid,
            url=get_url(meme.caption, meme.templateid)
        )
        return AddMemeMutation(meme=new_meme)


class MemeMutation(graphene.Mutation):
    class Arguments:
        id = graphene.Int(required=True)
        caption = graphene.String()
        templateid = graphene.String()
    meme = graphene.Field(MemeType)

    def mutate(self, info, id, caption=None, templateid=None):
        meme = Meme.objects.get(pk=id)
        if caption is not None:
            meme.caption = caption
        if templateid is not None:
            meme.templateid = templateid
        meme.save()
        return MemeMutation(meme=meme)


class DeleteMemeMutation(graphene.Mutation):
    class Arguments:
        id = graphene.Int(required=True)
    ok = graphene.Boolean()

    def mutate(self, info, **kwargs):
        meme = Meme.objects.filter(pk=kwargs.get("id")).first()
        count, _ = meme.delete()
        deleted = count == 1
        return DeleteMemeMutation(ok=deleted)


class Mutation(graphene.ObjectType):
    edit_meme = MemeMutation.Field()
    add_meme = AddMemeMutation.Field()
    delete_meme = DeleteMemeMutation.Field()


schema = graphene.Schema(query=Query, mutation=Mutation)
