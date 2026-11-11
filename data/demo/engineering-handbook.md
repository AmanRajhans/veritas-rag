# Northstar Engineering Handbook

## Incident response

The on-call engineer must acknowledge a severity-one page within five minutes. They become incident commander until they explicitly hand off the role. The commander opens an incident channel, assigns a scribe, and posts a customer-facing update within thirty minutes. Severity-two incidents have a fifteen-minute acknowledgement target and do not require a public update unless customer impact lasts longer than one hour.

After service is restored, the incident owner schedules a blameless review within three business days. The written review must include a timeline, contributing factors, customer impact, and corrective actions with named owners. Corrective actions are tracked in the reliability backlog; severity-one actions are reviewed weekly by engineering leadership until closed.

## Deployments

Production changes require a passing continuous-integration build, one approving reviewer, and a linked change record. Deployments are allowed Monday through Thursday between 09:00 and 15:00 Pacific Time. A release outside that window needs approval from the on-call engineering manager and a documented rollback plan.

Every deployment uses a canary stage receiving five percent of traffic for at least ten minutes. The release automatically rolls back when the five-minute error rate exceeds two percent or p95 latency rises by more than thirty percent from the pre-deployment baseline. Engineers may manually promote only after checking error rate, latency, and saturation dashboards.

## Access control

Production access follows least privilege and is granted for at most ninety days. Requests require the employee's manager and the system owner to approve. Break-glass access expires after four hours, creates a security alert, and requires a retrospective review by the next business day. Shared accounts are prohibited.

## Data retention

Application logs are retained for thirty days in the searchable tier and an additional sixty days in encrypted archive storage. Security audit logs are retained for one year. Customer deletion requests must propagate to primary systems within seven days and to backups within thirty-five days.
