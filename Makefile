all:
	echo "no default recipe"

check_availability:
	aws ec2 describe-availability-zones --region us-west-2 --output text

create_reserved_instance:
	aws ec2 run-instances \
		--image-id ami-6e1a0117 \
		--instance-type t2.micro \
		--key-name aws_moctezuma_0 \
		--count 1 \
		--dry-run \
		> aws/instance-details.json
		#--block-device-mappings '[{"DeviceName":"/dev/xvdf","Ebs":{"SnapshotId":"snap-xxxxxxxx"}}]'

create_spot_instance:
	aws ec2 request-spot-instances \
		--client-token drammock_moctezuma_eegphonecoding \
		--instance-count 1 \
		--spot-price "1.50" \
		--type "one-time" \
		--launch-specification file://aws/launch-spec.json \
		--dry-run \
		> aws/instance-details.json

get_instance_info:
	aws ec2 describe-instances > aws/instance-details.json
	jq '.Reservations[] | .Instances[] | .InstanceId'                 aws/instance-details.json | tr -d '"' > aws/instance-id.txt
	jq '.Reservations[] | .Instances[] | .Placement.AvailabilityZone' aws/instance-details.json | tr -d '"' > aws/availability-zone.txt
	jq '.Reservations[] | .Instances[] | .PublicDnsName'              aws/instance-details.json | tr -d '"' > aws/ip-address.txt
	cat aws/instance-id.txt
	cat aws/availability-zone.txt
	cat aws/ip-address.txt

create_ebs:
	aws ec2 create-volume \
		--size 50 \
		--region us-west-2 \
		--availability-zone $$(cat aws/availability-zone.txt) \
		--volume-type gp2 \
		> aws/ebs-details.json

get_ebs_info:
	jq '.VolumeId' aws/ebs-details.json | tr -d '"' > aws/volume-id.txt
	cat aws/volume-id.txt

attach_ebs:
	aws ec2 attach-volume \
		--volume-id $$(cat aws/volume-id.txt) \
		--instance-id $$(cat aws/instance-id.txt) \
		--device /dev/xvdf \
		--region us-west-2 \
		> aws/attachment-details.json

transfer:
	rsync -e "ssh -i ~/.ssh/aws_moctezuma_0.pem" -vh --progress \
		--files-from=aws/aws-rsync-list.txt . \
		ubuntu@$$(cat aws/ip-address.txt):~/ebsdrive/eeg_phone_coding

connect:
	ssh -i ~/.ssh/aws_moctezuma_0.pem ubuntu@$$(cat aws/ip-address.txt)

download:
	rsync -e "ssh -i ~/.ssh/aws_moctezuma_0.pem" -vhr --progress \
		ubuntu@$$(cat aws/ip-address.txt):~/ebsdrive/eeg_phone_coding/processed-data/ \
		processed-data

del_volume:
	aws ec2 delete-volume --volume-id $$(cat aws/volume-id.txt)

terminate:
	aws ec2 terminate-instances --instance-ids $$(cat aws/instance-id.txt)
