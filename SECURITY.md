# Security

Do not commit AWS credentials, access tokens, private keys, `.env` files, model
weights, datasets or inference media to this repository.

The AWS account identifiers, IAM role names and immutable ECR image digests in
the deployment manifests identify deployment resources; they are not
credentials. Authentication must come from an approved AWS profile, SSO session
or workload role.

Before every pull request, scan the staged diff for secrets and confirm that no
large binary artifact has been added.
