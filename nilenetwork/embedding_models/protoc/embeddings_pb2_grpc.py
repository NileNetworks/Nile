# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import Nilenetworks.embedding_models.protoc.embeddings_pb2 as embeddings__pb2


class EmbeddingStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Embed = channel.unary_unary(
            "/Embedding/Embed",
            request_serializer=embeddings__pb2.EmbeddingRequest.SerializeToString,
            response_deserializer=embeddings__pb2.BatchEmbeds.FromString,
        )


class EmbeddingServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Embed(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_EmbeddingServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "Embed": grpc.unary_unary_rpc_method_handler(
            servicer.Embed,
            request_deserializer=embeddings__pb2.EmbeddingRequest.FromString,
            response_serializer=embeddings__pb2.BatchEmbeds.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "Embedding", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class Embedding(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Embed(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/Embedding/Embed",
            embeddings__pb2.EmbeddingRequest.SerializeToString,
            embeddings__pb2.BatchEmbeds.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
